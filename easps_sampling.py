import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class EASPSSamplingConfig:
    max_new_tokens: int = 2048
    alpha: float = 4.0
    alpha_exp_k: float = 1.0
    n_particles: int = 16
    ess_threshold: float = 0.5
    temperature: float = 0.25
    block_size: int = 64
    force_eos_after_done: bool = True
    min_new_tokens: int = 100
    repetition_penalty: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_tokens_to_keep: int = 1
    penalize_prompt: bool = False
    hard_truncation: bool = True
    soft_truncation_value: float = -50.0
    stop_on_boxed: bool = True
    boxed_check_window_tokens: int = 256
    use_cow_cache: bool = True
    shared_prompt_cache: bool = True
    compact_done_cache: bool = True
    seed: int = 0
    empty_cache_on_resample: bool = False
    collect_telemetry: bool = False
    multiround_combine: str = "logz"
    debug_all: bool = False
    debug_verbose: bool = False
    debug_every: int = 32
    debug_particles: int = 4
    debug_text_every: int = 0
    debug_text_chars: int = 160
    debug_final_text: bool = True
    alpha_ramp_tokens: int = 400

    def __post_init__(self):
        for name in ("max_new_tokens", "n_particles", "block_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.alpha) or self.alpha < 1.0:
            raise ValueError("alpha must be finite and >= 1.0")
        if not math.isfinite(self.alpha_exp_k) or self.alpha_exp_k < 0.0:
            raise ValueError("alpha_exp_k must be finite and >= 0")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and > 0")
        if (
            not math.isfinite(self.ess_threshold)
            or not 0.0 <= self.ess_threshold <= 1.0
        ):
            raise ValueError("ess_threshold must be in [0, 1]")


EXP_FORMULA_VERSION = "h_no_progress_v1"
_DBG = "[smc]"
_BOX_ONLY_RE = re.compile("^[\\s{}]*$")


def _exp_schedule(t, alpha_final, ramp_T, entropy_mean, exp_k):
    alpha_t, exp_decay = _exponential_alpha(alpha_final, entropy_mean, exp_k)
    progress = min(1.0, float(t + 1) / float(max(int(ramp_T), 1)))
    alpha_linear = 1.0 + (float(alpha_final) - 1.0) * progress
    return (
        alpha_t,
        None,
        {
            "entropy_mean": max(float(entropy_mean), 0.0),
            "entropy_ema": max(float(entropy_mean), 0.0),
            "entropy_ratio": exp_decay,
            "curve_power": 1.0,
            "time_progress": progress,
            "adapted_progress": exp_decay,
            "alpha_linear": alpha_linear,
            "exp_k": float(exp_k),
            "exp_decay": exp_decay,
        },
    )


def _resampling_decision(*, ess, n_particles, ess_threshold):
    fixed_ratio = float(ess_threshold)
    fixed_threshold = fixed_ratio * int(n_particles)
    return {
        "will_resample": float(ess) < fixed_threshold,
        "ess_threshold": fixed_threshold,
        "adaptive_ess_ratio": fixed_ratio,
        "ess_eff": float(ess),
        "entropy_mean": None,
        "entropy_ema_ref": None,
        "entropy_ratio": None,
        "entropy_ema": None,
        "alpha_next": None,
        "ess_pred": None,
        "g_mean": None,
        "g_var": None,
        "branching_score": None,
    }


def _exponential_alpha(alpha_final, entropy_mean, exp_k):
    if not math.isfinite(float(entropy_mean)):
        raise ValueError("entropy_mean must be finite")
    entropy_value = max(float(entropy_mean), 0.0)
    exp_decay = math.exp(-float(exp_k) * entropy_value)
    return (1.0 + (float(alpha_final) - 1.0) * exp_decay, exp_decay)


def apply_repetition_penalty_(logits, prev_tokens, penalty):
    if penalty is None or penalty == 1.0:
        return logits
    N, V = logits.shape
    for i in range(N):
        toks = prev_tokens[i]
        if toks.numel() == 0:
            continue
        uniq = torch.unique(toks)
        row = logits[i]
        vals = row.index_select(0, uniq)
        vals = torch.where(vals < 0, vals * penalty, vals / penalty)
        row.index_copy_(0, uniq, vals)
    return logits


def top_k_filter_(logits, top_k):
    if top_k is None or top_k <= 0 or top_k >= logits.size(-1):
        return logits
    kth_vals = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
    logits.masked_fill_(logits < kth_vals, -float("inf"))
    return logits


def top_p_filter_(logits, top_p, min_tokens_to_keep=1):
    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)
    sorted_mask = cumprobs > top_p
    if min_tokens_to_keep > 1:
        sorted_mask[:, :min_tokens_to_keep] = False
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_idx, src=sorted_mask)
    logits.masked_fill_(mask, -float("inf"))
    return logits


def effective_sample_size(w, eps=1e-12):
    w = w.clamp_min(eps)
    return float(1.0 / torch.sum(w * w).item())


def systematic_resample(w, generator=None):
    N = w.numel()
    device = w.device
    if generator is None:
        u0 = torch.rand((), device=device)
    else:
        u0 = torch.rand((), device=device, generator=generator)
    positions = (u0 + torch.arange(N, device=device)) / N
    cdf = torch.cumsum(w, dim=0)
    cdf[-1] = 1.0
    idx = torch.searchsorted(cdf, positions, right=False)
    idx = idx.clamp_max(N - 1).to(torch.long)
    return idx


def _uses_phi_dynamic_cache(model) -> bool:
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    return str(model_type).strip().lower() == "phi3"


def _forward_prompt(model, input_ids: torch.Tensor):
    if not _uses_phi_dynamic_cache(model):
        return model(input_ids=input_ids, use_cache=True)
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:
        raise RuntimeError(
            "Phi-3 sampling requires transformers.cache_utils.DynamicCache"
        ) from exc
    prompt_cache = DynamicCache()
    out = model(input_ids=input_ids, past_key_values=prompt_cache, use_cache=True)
    if not isinstance(getattr(out, "past_key_values", None), DynamicCache):
        raise RuntimeError("Phi-3 returned a legacy cache despite DynamicCache prefill")
    return out


def _reorder_tensor_batchdim(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return t.index_select(0, idx.to(t.device))


def _recursive_reorder(obj, idx):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if obj.dim() >= 1 and obj.size(0) == idx.numel():
            return _reorder_tensor_batchdim(obj, idx)
        return obj
    if isinstance(obj, tuple):
        return tuple((_recursive_reorder(x, idx) for x in obj))
    if isinstance(obj, list):
        return [_recursive_reorder(x, idx) for x in obj]
    if isinstance(obj, dict):
        return {k: _recursive_reorder(v, idx) for k, v in obj.items()}
    return obj


def reorder_past_key_values(model, past_key_values, idx):
    if past_key_values is None:
        return None
    if (
        _uses_phi_dynamic_cache(model)
        and hasattr(past_key_values, "reorder_cache")
        and callable(getattr(past_key_values, "reorder_cache"))
    ):
        try:
            past_key_values.reorder_cache(idx)
            return past_key_values
        except Exception:
            pass
    if hasattr(model, "_reorder_cache") and callable(getattr(model, "_reorder_cache")):
        try:
            return model._reorder_cache(past_key_values, idx)
        except Exception:
            pass
    if hasattr(past_key_values, "reorder_cache") and callable(
        getattr(past_key_values, "reorder_cache")
    ):
        try:
            past_key_values.reorder_cache(idx)
            return past_key_values
        except Exception:
            pass
    return _recursive_reorder(past_key_values, idx)


def _recursive_select_batch(obj, idx):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if idx.numel() == 0:
            return obj
        if obj.dim() >= 1 and obj.size(0) >= idx.max().item() + 1:
            return obj.index_select(0, idx.to(obj.device)).contiguous()
        return obj
    if isinstance(obj, tuple):
        return tuple((_recursive_select_batch(x, idx) for x in obj))
    if isinstance(obj, list):
        return [_recursive_select_batch(x, idx) for x in obj]
    if isinstance(obj, dict):
        return {k: _recursive_select_batch(v, idx) for k, v in obj.items()}
    return obj


def _recursive_expand_batch(obj, expand_idx):
    if obj is None:
        return None
    if torch.is_tensor(obj):
        if expand_idx.numel() == 0:
            return obj
        U = int(expand_idx.max().item()) + 1
        if obj.dim() >= 1 and obj.size(0) == U:
            return obj.index_select(0, expand_idx.to(obj.device))
        return obj
    if isinstance(obj, tuple):
        return tuple((_recursive_expand_batch(x, expand_idx) for x in obj))
    if isinstance(obj, list):
        return [_recursive_expand_batch(x, expand_idx) for x in obj]
    if isinstance(obj, dict):
        return {k: _recursive_expand_batch(v, expand_idx) for k, v in obj.items()}
    return obj


def select_cache_subset(model, past, idx):
    if past is None:
        return None
    if hasattr(past, "reorder_cache") and callable(getattr(past, "reorder_cache")):
        try:
            past.reorder_cache(idx)
            return past
        except Exception:
            pass
    if hasattr(model, "_reorder_cache") and callable(getattr(model, "_reorder_cache")):
        try:
            return model._reorder_cache(past, idx)
        except Exception:
            pass
    return _recursive_select_batch(past, idx)


def expand_cache(model, past, expand_idx):
    if past is None:
        return None
    if hasattr(past, "reorder_cache") and callable(getattr(past, "reorder_cache")):
        try:
            past.reorder_cache(expand_idx)
            return past
        except Exception:
            pass
    if hasattr(model, "_reorder_cache") and callable(getattr(model, "_reorder_cache")):
        try:
            return model._reorder_cache(past, expand_idx)
        except Exception:
            pass
    return _recursive_expand_batch(past, expand_idx)


def _logical_logits_from_physical(
    physical_logits: torch.Tensor, logical_to_physical: torch.Tensor
) -> torch.Tensor:
    if logical_to_physical.dim() != 1:
        raise ValueError("logical_to_physical must be one-dimensional")
    live = logical_to_physical >= 0
    if not bool(live.any().item()):
        raise ValueError("at least one logical particle must have a physical cache")
    safe_idx = logical_to_physical.clone()
    safe_idx[~live] = logical_to_physical[live][0]
    if int(safe_idx.max().item()) >= int(physical_logits.size(0)):
        raise ValueError("logical_to_physical references a missing physical row")
    return physical_logits.index_select(0, safe_idx)


def _select_physical_cache_rows(model, past, idx, n_physical):
    identity = torch.arange(idx.numel(), dtype=torch.long, device=idx.device)
    if int(idx.numel()) == int(n_physical) and torch.equal(idx, identity):
        return past
    return select_cache_subset(model, past, idx)


def _compact_cache_for_decode(
    model,
    past,
    logical_to_physical: torch.Tensor,
    done: torch.Tensor,
    *,
    n_physical: int,
    active_only: bool,
):
    keep = ~done if active_only else torch.ones_like(done, dtype=torch.bool)
    decode_logical_idx = torch.nonzero(keep, as_tuple=False).flatten()
    if decode_logical_idx.numel() == 0:
        raise ValueError("cannot decode an empty active-particle batch")
    source_idx = logical_to_physical.index_select(0, decode_logical_idx)
    if bool((source_idx < 0).any().item()):
        raise ValueError("an active logical particle has no physical cache")
    past = _select_physical_cache_rows(model, past, source_idx, n_physical)
    new_mapping = torch.full_like(logical_to_physical, -1)
    new_mapping[decode_logical_idx] = torch.arange(
        decode_logical_idx.numel(), dtype=torch.long, device=done.device
    )
    return (past, decode_logical_idx, new_mapping)


def _resample_compacted_cache(
    model,
    past,
    physical_logits: torch.Tensor,
    logical_to_physical: torch.Tensor,
    done: torch.Tensor,
    idx_rs: torch.Tensor,
    *,
    use_cow_cache: bool,
    active_only: bool,
):
    resampled_done = done.index_select(0, idx_rs)
    keep = (
        ~resampled_done
        if active_only
        else torch.ones_like(resampled_done, dtype=torch.bool)
    )
    kept_logical_idx = torch.nonzero(keep, as_tuple=False).flatten()
    no_active_children = kept_logical_idx.numel() == 0
    if no_active_children:
        selected_idx = torch.zeros(1, dtype=torch.long, device=idx_rs.device)
        mapped = torch.empty(0, dtype=torch.long, device=idx_rs.device)
    else:
        ancestor_logical_idx = idx_rs.index_select(0, kept_logical_idx)
        source_idx = logical_to_physical.index_select(0, ancestor_logical_idx)
        if bool((source_idx < 0).any().item()):
            raise ValueError("resampling requested a missing active ancestor cache")
        if use_cow_cache:
            selected_idx, mapped = torch.unique(source_idx, return_inverse=True)
        else:
            selected_idx = source_idx
            mapped = torch.arange(
                source_idx.numel(), dtype=torch.long, device=source_idx.device
            )
    old_n_physical = int(physical_logits.size(0))
    past = _select_physical_cache_rows(model, past, selected_idx, old_n_physical)
    physical_logits = physical_logits.index_select(0, selected_idx)
    new_mapping = torch.full_like(logical_to_physical, -1)
    new_mapping[kept_logical_idx] = mapped
    if no_active_children:
        new_mapping[0] = 0
    return (past, physical_logits, new_mapping)


def _build_prompt_cache(model, input_ids_1: torch.Tensor, N: int):
    out = _forward_prompt(model, input_ids_1)
    past_1 = getattr(out, "past_key_values", None)
    logits_1 = out.logits[:, -1, :]
    expand_idx = torch.zeros(N, dtype=torch.long, device=input_ids_1.device)
    past_N = expand_cache(model, past_1, expand_idx)
    logits_N = logits_1.expand(N, -1).contiguous()
    del out
    torch.cuda.empty_cache()
    return (past_N, logits_N)


def _resolve_eos_token_ids(model, tokenizer) -> Tuple[int, ...]:
    generation_config = getattr(model, "generation_config", None)
    raw_ids = getattr(generation_config, "eos_token_id", None)
    if raw_ids is None:
        raw_ids = getattr(getattr(model, "config", None), "eos_token_id", None)
    values = []
    for raw in (raw_ids, getattr(tokenizer, "eos_token_id", None)):
        if raw is None:
            continue
        if torch.is_tensor(raw):
            candidates = raw.detach().flatten().tolist()
        elif isinstance(raw, (list, tuple, set)):
            candidates = raw
        else:
            candidates = (raw,)
        for value in candidates:
            value = int(value)
            if value not in values:
                values.append(value)
    if not values:
        raise ValueError("An eos_token_id is required on the model or tokenizer.")
    return tuple(values)


def _logsumexp(x, dim=0):
    return torch.logsumexp(x, dim=dim)


def _entropy_from_logprobs(logprobs: torch.Tensor) -> torch.Tensor:
    probs = logprobs.exp()
    terms = torch.where(probs > 0, -probs * logprobs, torch.zeros_like(probs))
    return terms.sum(dim=-1)


def _has_nonempty_boxed(text):
    for macro in ("\\boxed", "\\fbox"):
        start = 0
        while True:
            idx = text.find(macro, start)
            if idx < 0:
                break
            j = idx + len(macro)
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] == "{":
                depth = 0
                right = None
                for k in range(j, len(text)):
                    c = text[k]
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            right = k
                            break
                if right is not None:
                    content = text[j + 1 : right].strip()
                    while (
                        len(content) >= 2 and content[0] == "{" and (content[-1] == "}")
                    ):
                        inner = content[1:-1].strip()
                        if inner == content:
                            break
                        content = inner
                    if content and _BOX_ONLY_RE.fullmatch(content) is None:
                        return True
            start = idx + len(macro)
    return False


def _fmt_mask(mask) -> str:
    if torch.is_tensor(mask):
        mask = mask.tolist()
    return "".join(("1" if int(v) else "0" for v in mask))


def _dbg_tag(round_idx):
    return _DBG if round_idx is None else f"[smc r{round_idx}]"


def _fmt_mult(idx_rs) -> str:
    vals, counts = torch.unique(idx_rs, return_counts=True)
    pairs = sorted(zip(vals.tolist(), counts.tolist()), key=lambda kv: (-kv[1], kv[0]))
    head = ", ".join((f"p{a}->{c}" for a, c in pairs[:8]))
    if len(pairs) > 8:
        head += f", ... (+{len(pairs) - 8})"
    return head


def _fmt_entropy_triplet(values) -> str:
    if values is None:
        return f"{'n/a':>8} {'n/a':>7} {'n/a':>7}"
    return f"{values.mean():>8.3f} {values.min():>7.3f} {values.max():>7.3f}"


def _tail_text(tokenizer, row_ids, start_tok, cur, n_chars) -> str:
    text = tokenizer.decode(row_ids[start_tok:cur].tolist(), skip_special_tokens=True)
    text = " ".join(text.split())
    return text[-n_chars:] if len(text) > n_chars else text


def _combine_round_log_w(all_log_w, all_log_z, mode):
    mode = str(mode).lower()
    if mode == "legacy":
        return torch.cat(all_log_w, dim=0)
    if mode == "logz":
        shifted = [lw + float(lz) for lw, lz in zip(all_log_w, all_log_z)]
        return torch.cat(shifted, dim=0)
    if mode == "mixture":
        n_r = len(all_log_w)
        log_r = math.log(max(n_r, 1))
        normed = [lw - _logsumexp(lw, dim=0) - log_r for lw in all_log_w]
        return torch.cat(normed, dim=0)
    raise ValueError(f"Unknown multiround_combine: {mode!r} (logz | mixture | legacy)")


@torch.no_grad()
def easps_power_sample_memopt(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cfg: EASPSSamplingConfig,
    round_idx: Optional[int] = None,
) -> Dict[str, Any]:
    assert input_ids.dim() == 2 and input_ids.size(0) == 1
    device = input_ids.device
    g = torch.Generator(device=device)
    g.manual_seed(int(cfg.seed))
    debug = bool(cfg.debug_verbose)
    telemetry = bool(cfg.collect_telemetry) or bool(cfg.debug_all) or debug
    tag = _dbg_tag(round_idx)
    entropy_signal = "none"
    alpha_bridge_schedule = "exp"
    compute_step_entropy = True
    N = int(cfg.n_particles)
    active_cache_compaction = bool(
        cfg.compact_done_cache
        and cfg.force_eos_after_done
        and _uses_phi_dynamic_cache(model)
    )
    prompt_len = input_ids.size(1)
    eos_ids = _resolve_eos_token_ids(model, tokenizer)
    eos_ids_tensor = torch.tensor(eos_ids, dtype=torch.long, device=device)
    eos_id = (
        int(tokenizer.eos_token_id)
        if tokenizer.eos_token_id is not None
        else eos_ids[0]
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    if cfg.shared_prompt_cache:
        past, last_logits = _build_prompt_cache(model, input_ids, N)
    else:
        prompt = input_ids.expand(N, -1).contiguous()
        out = _forward_prompt(model, prompt)
        past = getattr(out, "past_key_values", None)
        last_logits = out.logits[:, -1, :]
    total_len = prompt_len + int(cfg.max_new_tokens)
    seqs = torch.full((N, total_len), pad_id, dtype=torch.long, device=device)
    seqs[:, :prompt_len] = input_ids.expand(N, -1)
    cur = prompt_len
    done = torch.zeros(N, dtype=torch.bool, device=device)
    cum_logp = torch.zeros(N, dtype=torch.float32, device=device)
    log_w = torch.zeros(N, dtype=torch.float32, device=device)
    log_z_partial = 0.0
    alpha_final = float(cfg.alpha)
    ramp_T = max(int(cfg.alpha_ramp_tokens), 1)
    prev_alpha = 1.0
    alpha_entropy_ema = None
    stats = {
        "method": "EASPS",
        "resample_count": 0,
        "ess_history": [],
        "mean_logw_history": [],
        "max_logw_history": [],
        "alpha_schedule": alpha_bridge_schedule,
        "alpha_bridge_schedule": alpha_bridge_schedule,
        "alpha_formula_version": EXP_FORMULA_VERSION,
        "proposal_from_alpha": bool(False),
        "alpha_exp_k": float(cfg.alpha_exp_k),
        "alpha_entropy_ema_beta": float(0.9),
        "alpha_edt_lambda": float(1.0),
        "alpha_cess_target": float(0.8),
        "alpha_branch_gamma": float(1.0),
        "resample_branch_lambda": float(0.5),
        "alpha_branching_history": [],
        "alpha_cess_t_history": [],
        "alpha_cess_ratio_history": [],
        "alpha_stage_history": [],
        "entropy_computed": bool(compute_step_entropy),
        "alpha_history": [],
        "alpha_entropy_mean_history": [],
        "alpha_entropy_ema_history": [],
        "alpha_entropy_ratio_history": [],
        "alpha_curve_power_history": [],
        "alpha_progress_history": [],
        "alpha_time_progress_history": [],
        "resample_branching_history": [],
        "resample_ess_pred_history": [],
        "adaptive_ess_ratio_history": [],
        "proposal_temperature_history": [],
        "unique_ancestors_history": [],
        "peak_batch_size_history": [],
        "decode_batch_size_history": [],
        "compact_done_cache": active_cache_compaction,
        "done_at": None,
    }
    Tmax = int(cfg.max_new_tokens)
    min_new = max(int(cfg.min_new_tokens), 0)
    entropy_hist = None
    surprisal_hist = None
    done_hist = None
    resample_mask = None
    block_mask = None
    if telemetry:
        if compute_step_entropy:
            entropy_hist = torch.zeros((N, Tmax), dtype=torch.float32, device=device)
        surprisal_hist = torch.zeros((N, Tmax), dtype=torch.float32, device=device)
        done_hist = torch.zeros((N, Tmax), dtype=torch.uint8, device=device)
        resample_mask = torch.zeros(Tmax, dtype=torch.uint8)
        block_mask = torch.zeros(Tmax, dtype=torch.uint8)
        stats["resample_events"] = []
        stats["block_history"] = []
    if debug:
        print(
            f"{tag} ===== start  N={N} prompt_len={prompt_len} max_new={Tmax} block={int(cfg.block_size)} ess_thr={float(cfg.ess_threshold)}(*N={float(cfg.ess_threshold) * N:.1f}) alpha={alpha_final} ramp={ramp_T} temp={float(cfg.temperature)} alpha_bridge_schedule={alpha_bridge_schedule} proposal={('alpha' if False else 'fixed')} signal={entropy_signal} =====",
            flush=True,
        )
        print(
            f"{tag} {'t':>5} {'cur':>5} {'done':>7} {'H(mean)':>8} {'H(min)':>7} {'H(max)':>7} {'surp':>7} {'logw':>9} {'alpha':>6} {'nphys':>5}",
            flush=True,
        )
    cow_active = False
    logical_to_physical = (
        torch.arange(N, dtype=torch.long, device=device)
        if active_cache_compaction
        else None
    )
    n_physical = N
    for t in range(Tmax):
        logits_phys = last_logits.float()
        base_logprobs_phys = F.log_softmax(logits_phys, dim=-1)
        if active_cache_compaction:
            raw_logits = _logical_logits_from_physical(logits_phys, logical_to_physical)
            base_logprobs = _logical_logits_from_physical(
                base_logprobs_phys, logical_to_physical
            )
        elif cow_active and logical_to_physical is not None:
            raw_logits = logits_phys[logical_to_physical]
            base_logprobs = base_logprobs_phys[logical_to_physical]
        else:
            raw_logits = logits_phys
            base_logprobs = base_logprobs_phys
        step_entropy = None
        alpha_entropy_mean = None
        alpha_branching_mean = None
        active = ~done
        if compute_step_entropy:
            step_entropy = _entropy_from_logprobs(base_logprobs)
            if bool(active.any().item()):
                alpha_entropy_mean = float(step_entropy[active].mean().item())
            else:
                alpha_entropy_mean = float(step_entropy.mean().item())
        alpha_t, alpha_entropy_ema, alpha_info = _exp_schedule(
            t=t,
            alpha_final=alpha_final,
            ramp_T=ramp_T,
            entropy_mean=alpha_entropy_mean,
            exp_k=cfg.alpha_exp_k,
        )
        stats["alpha_history"].append(float(alpha_t))
        stats["alpha_branching_history"].append(
            None if alpha_branching_mean is None else float(alpha_branching_mean)
        )
        stats["alpha_entropy_mean_history"].append(
            None if alpha_entropy_mean is None else float(alpha_entropy_mean)
        )
        stats["alpha_entropy_ema_history"].append(
            None if alpha_entropy_ema is None else float(alpha_entropy_ema)
        )
        stats["alpha_entropy_ratio_history"].append(float(alpha_info["entropy_ratio"]))
        stats["alpha_curve_power_history"].append(float(alpha_info["curve_power"]))
        stats["alpha_progress_history"].append(float(alpha_info["adapted_progress"]))
        stats["alpha_time_progress_history"].append(float(alpha_info["time_progress"]))
        proposal_temperature = float(cfg.temperature)
        if proposal_temperature == 1.0:
            prop_logits = raw_logits.clone()
        else:
            prop_logits = raw_logits / proposal_temperature
        stats["proposal_temperature_history"].append(proposal_temperature)
        if t < min_new:
            prop_logits[:, list(eos_ids)] = -float("inf")
        if cfg.repetition_penalty is not None and cfg.repetition_penalty != 1.0:
            if cfg.penalize_prompt:
                history = seqs[:, :cur]
            else:
                history = seqs[:, prompt_len:cur]
            apply_repetition_penalty_(prop_logits, history, cfg.repetition_penalty)
        if cfg.top_k is not None and cfg.top_k > 0:
            top_k_filter_(prop_logits, cfg.top_k)
        if cfg.top_p is not None and cfg.top_p < 1.0:
            top_p_filter_(
                prop_logits, cfg.top_p, min_tokens_to_keep=cfg.min_tokens_to_keep
            )
        if not cfg.hard_truncation:
            prop_logits = torch.where(
                torch.isinf(prop_logits),
                torch.full_like(prop_logits, float(cfg.soft_truncation_value)),
                prop_logits,
            )
        prop_logprobs = F.log_softmax(prop_logits, dim=-1)
        if telemetry:
            if entropy_hist is not None:
                entropy_hist[:, t] = step_entropy
            done_hist[:, t] = done.to(torch.uint8)
        next_tokens = torch.empty(N, dtype=torch.long, device=device)
        if cfg.force_eos_after_done:
            active = ~done
            if active.any():
                probs_active = torch.exp(prop_logprobs[active])
                next_tokens[active] = torch.multinomial(
                    probs_active, num_samples=1, generator=g
                ).squeeze(-1)
            next_tokens[done] = eos_id
        else:
            probs = torch.exp(prop_logprobs)
            next_tokens = torch.multinomial(probs, num_samples=1, generator=g).squeeze(
                -1
            )
        idx_tok = next_tokens.view(N, 1)
        token_logp = torch.gather(base_logprobs, dim=-1, index=idx_tok).squeeze(-1)
        token_logq = torch.gather(prop_logprobs, dim=-1, index=idx_tok).squeeze(-1)
        if telemetry:
            surprisal_hist[:, t] = -token_logp
        if cfg.force_eos_after_done:
            token_logp = torch.where(done, torch.zeros_like(token_logp), token_logp)
            token_logq = torch.where(done, torch.zeros_like(token_logq), token_logq)
        cum_logp_prev = cum_logp
        delta = float(alpha_t - prev_alpha)
        if delta != 0.0:
            log_w = log_w + delta * cum_logp_prev
            prev_alpha = float(alpha_t)
        log_w = log_w + (float(alpha_t) - 1.0) * token_logp
        log_w = log_w + (token_logp - token_logq)
        cum_logp = cum_logp + token_logp
        seqs[:, cur] = next_tokens
        cur += 1
        newly_done = torch.isin(next_tokens, eos_ids_tensor) & ~done
        done = done | newly_done
        if cfg.stop_on_boxed:
            active = ~done
            if active.any():
                start_tok = max(prompt_len, cur - int(cfg.boxed_check_window_tokens))
                active_idx = torch.nonzero(active, as_tuple=False).flatten().tolist()
                for i_part in active_idx:
                    gen_text_tail = tokenizer.decode(
                        seqs[i_part, start_tok:cur].tolist(), skip_special_tokens=True
                    )
                    if _has_nonempty_boxed(gen_text_tail):
                        done[i_part] = True
        if stats["done_at"] is None and done.all():
            stats["done_at"] = t + 1
        is_block_end_now = (t + 1) % int(cfg.block_size) == 0 or t == Tmax - 1
        if debug and (not is_block_end_now):
            every = max(int(cfg.debug_every), 0)
            if every and t % every == 0:
                ent_t = None if entropy_hist is None else entropy_hist[:, t]
                print(
                    f"{tag} {t:>5} {cur:>5} {int(done.sum()):>3}/{N:<3} {_fmt_entropy_triplet(ent_t)} {surprisal_hist[:, t].mean():>7.3f} {log_w.mean():>9.2f} {alpha_t:>6.2f} {n_physical:>5}",
                    flush=True,
                )
        if (
            debug
            and int(cfg.debug_text_every) > 0
            and (t % int(cfg.debug_text_every) == 0)
        ):
            n_show = min(int(cfg.debug_particles), N)
            start_tok = max(prompt_len, cur - 4 * int(cfg.debug_text_chars))
            print(
                f"{tag}   -- text tail @t={t} (particle 0..{n_show - 1}) --", flush=True
            )
            for i in range(n_show):
                flag = "DONE" if bool(done[i]) else "    "
                h_t = "n/a" if entropy_hist is None else f"{entropy_hist[i, t]:.3f}"
                print(
                    f"{tag}   p{i:<3}{flag} H={h_t} | ...{_tail_text(tokenizer, seqs[i], start_tok, cur, int(cfg.debug_text_chars))}",
                    flush=True,
                )
        if done.all():
            break
        if active_cache_compaction:
            past, decode_logical_idx, logical_to_physical = _compact_cache_for_decode(
                model,
                past,
                logical_to_physical,
                done,
                n_physical=n_physical,
                active_only=True,
            )
            n_physical = int(decode_logical_idx.numel())
            stats["decode_batch_size_history"].append(n_physical)
            out = model(
                input_ids=next_tokens.index_select(0, decode_logical_idx).view(-1, 1),
                past_key_values=past,
                use_cache=True,
            )
        else:
            if cow_active and logical_to_physical is not None:
                past = expand_cache(model, past, logical_to_physical)
                cow_active = False
                logical_to_physical = None
                n_physical = N
            out = model(
                input_ids=next_tokens.view(N, 1), past_key_values=past, use_cache=True
            )
        past = getattr(out, "past_key_values", None)
        last_logits = out.logits[:, -1, :]
        is_block_end = (t + 1) % int(cfg.block_size) == 0 or t == Tmax - 1
        if is_block_end:
            lw = log_w - _logsumexp(log_w, dim=0)
            w = torch.exp(lw)
            ess = effective_sample_size(w)
            stats["ess_history"].append(ess)
            stats["mean_logw_history"].append(float(log_w.mean().item()))
            stats["max_logw_history"].append(float(log_w.max().item()))
            decision = _resampling_decision(
                ess=ess, n_particles=N, ess_threshold=cfg.ess_threshold
            )
            will_resample = bool(decision["will_resample"])
            ess_thr = float(decision["ess_threshold"])
            adaptive_ess_ratio = float(decision["adaptive_ess_ratio"])
            ess_eff = float(decision["ess_eff"])
            entropy_mean_t = decision["entropy_mean"]
            entropy_ema_ref = decision["entropy_ema_ref"]
            entropy_ratio = decision["entropy_ratio"]
            alpha_next = decision["alpha_next"]
            ess_pred = decision["ess_pred"]
            g_mean = decision["g_mean"]
            g_var = decision["g_var"]
            branching_score = decision["branching_score"]
            stats["resample_branching_history"].append(branching_score)
            stats["resample_ess_pred_history"].append(
                None if ess_pred is None else float(ess_pred)
            )
            stats["adaptive_ess_ratio_history"].append(float(adaptive_ess_ratio))
            if telemetry:
                block_mask[t] = 1
                resample_mask[t] = 1 if will_resample else 0
                ent_t = None if entropy_hist is None else entropy_hist[:, t]
                stats["block_history"].append(
                    {
                        "t": int(t),
                        "cur": int(cur),
                        "ess": float(ess),
                        "ess_ratio": float(ess / N),
                        "ess_threshold": float(ess_thr),
                        "ess_threshold_max": float(cfg.ess_threshold),
                        "adaptive_ess_ratio": float(adaptive_ess_ratio),
                        "resampled": bool(will_resample),
                        "done": int(done.sum().item()),
                        "entropy_mean": (
                            None if entropy_mean_t is None else float(entropy_mean_t)
                        ),
                        "entropy_ema_ref": (
                            None if entropy_ema_ref is None else float(entropy_ema_ref)
                        ),
                        "entropy_ratio": (
                            None if entropy_ratio is None else float(entropy_ratio)
                        ),
                        "entropy_signal": entropy_signal,
                        "alpha_next": None if alpha_next is None else float(alpha_next),
                        "alpha_cess_target": None,
                        "alpha_stage": None,
                        "alpha_cess_ratio": None,
                        "ess_pred": None if ess_pred is None else float(ess_pred),
                        "ess_eff": float(ess_eff),
                        "g_mean": None if g_mean is None else float(g_mean),
                        "g_var": None if g_var is None else float(g_var),
                        "branching_score": (
                            None if branching_score is None else float(branching_score)
                        ),
                        "logw_mean": float(log_w.mean().item()),
                    }
                )
            if debug:
                ent_t = None if entropy_hist is None else entropy_hist[:, t]
                verdict = "RESAMPLE" if will_resample else "keep    "
                signal_value = (
                    branching_score
                    if entropy_signal in {"branch", "branch_lookahead"}
                    else entropy_mean_t
                )
                signal_text = "n/a" if signal_value is None else f"{signal_value:.3f}"
                extra = (
                    f" ess_pred={ess_pred:6.2f}"
                    if ess_pred is not None and math.isfinite(ess_pred)
                    else ""
                )
                print(
                    f"{tag} {t:>5} {cur:>5} {int(done.sum()):>3}/{N:<3} {_fmt_entropy_triplet(ent_t)} {surprisal_hist[:, t].mean():>7.3f} {log_w.mean():>9.2f} {alpha_t:>6.2f} {n_physical:>5}   | BLOCK ess={ess:6.2f}{extra} thr={ess_thr:6.2f} sig[{entropy_signal}]={signal_text} -> {verdict}",
                    flush=True,
                )
            if will_resample:
                idx_rs = systematic_resample(w, generator=g)
                if active_cache_compaction:
                    past, last_logits, logical_to_physical = _resample_compacted_cache(
                        model,
                        past,
                        last_logits,
                        logical_to_physical,
                        done,
                        idx_rs,
                        use_cow_cache=bool(cfg.use_cow_cache),
                        active_only=True,
                    )
                    n_physical = int(last_logits.size(0))
                    seqs = seqs.index_select(0, idx_rs)
                    done = done.index_select(0, idx_rs)
                    cum_logp = cum_logp.index_select(0, idx_rs)
                    stats["unique_ancestors_history"].append(
                        int(torch.unique(idx_rs).numel()) if cfg.use_cow_cache else N
                    )
                elif cfg.use_cow_cache:
                    unique_ancestors, inverse = torch.unique(
                        idx_rs, return_inverse=True
                    )
                    U = unique_ancestors.numel()
                    stats["unique_ancestors_history"].append(U)
                    seqs = seqs.index_select(0, idx_rs)
                    done = done.index_select(0, idx_rs)
                    cum_logp = cum_logp.index_select(0, idx_rs)
                    past = select_cache_subset(model, past, unique_ancestors)
                    last_logits = last_logits.index_select(0, unique_ancestors)
                    logical_to_physical = inverse
                    cow_active = True
                    n_physical = U
                else:
                    seqs = seqs.index_select(0, idx_rs)
                    done = done.index_select(0, idx_rs)
                    cum_logp = cum_logp.index_select(0, idx_rs)
                    past = reorder_past_key_values(model, past, idx_rs)
                    last_logits = last_logits.index_select(0, idx_rs)
                    stats["unique_ancestors_history"].append(N)
                if telemetry:
                    if entropy_hist is not None:
                        entropy_hist = entropy_hist.index_select(0, idx_rs)
                    surprisal_hist = surprisal_hist.index_select(0, idx_rs)
                    done_hist = done_hist.index_select(0, idx_rs)
                    mult = torch.bincount(idx_rs, minlength=N)
                    clone_mask = (mult.index_select(0, idx_rs) > 1).to(torch.uint8)
                    stats["resample_events"].append(
                        {
                            "t": int(t),
                            "cur": int(cur),
                            "ess": float(ess),
                            "ess_threshold": float(ess_thr),
                            "unique_ancestors": int(torch.unique(idx_rs).numel()),
                            "ancestors": idx_rs.tolist(),
                            "clone_mask": _fmt_mask(clone_mask),
                            "survived_mask": _fmt_mask((mult > 0).to(torch.uint8)),
                        }
                    )
                if debug:
                    print(
                        f"{tag}   -> resample #{stats['resample_count'] + 1}: uniq={int(torch.unique(idx_rs).numel())}/{N} survived={_fmt_mask((torch.bincount(idx_rs, minlength=N) > 0).to(torch.uint8))}",
                        flush=True,
                    )
                    print(f"{tag}      mult: {_fmt_mult(idx_rs)}", flush=True)
                log_z_partial += float(_logsumexp(log_w, dim=0).item()) - math.log(N)
                log_w = torch.zeros_like(log_w)
                stats["resample_count"] += 1
                stats["peak_batch_size_history"].append(n_physical)
                if cfg.empty_cache_on_resample and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    stats["alpha_before_final_catchup"] = float(prev_alpha)
    if prev_alpha != alpha_final:
        log_w = log_w + float(alpha_final - prev_alpha) * cum_logp
        prev_alpha = alpha_final
    final_lw = log_w - _logsumexp(log_w, dim=0)
    final_w = torch.exp(final_lw)
    stats["final_alpha"] = float(prev_alpha)
    stats["final_ess"] = effective_sample_size(final_w)
    seqs_out = seqs[:, :cur]
    lw = final_lw
    w = final_w
    chosen_idx = int(torch.multinomial(w, 1, generator=g).item())
    chosen_sequence = seqs_out[chosen_idx]
    entropy_np = None
    if telemetry:
        t_gen = cur - prompt_len
        entropy_np = (
            entropy_hist[:, :t_gen].cpu().numpy() if entropy_hist is not None else None
        )
        surprisal_np = surprisal_hist[:, :t_gen].cpu().numpy()
        done_np = done_hist[:, :t_gen].cpu().numpy()
        active = done_np == 0
        stats["n_particles"] = int(N)
        stats["t_gen"] = int(t_gen)
        stats["ess_threshold"] = float(cfg.ess_threshold)
        stats["entropy_signal"] = entropy_signal
        stats["block_size"] = int(cfg.block_size)
        stats["resample_mask"] = _fmt_mask(resample_mask[:t_gen])
        stats["block_mask"] = _fmt_mask(block_mask[:t_gen])
        stats["resample_t"] = torch.nonzero(resample_mask[:t_gen]).flatten().tolist()
        stats["surprisal_mean"] = (
            float(surprisal_np[active].mean()) if active.any() else 0.0
        )
        if entropy_np is None:
            stats["entropy_mean"] = None
            stats["entropy_std"] = None
            stats["entropy_max"] = None
            stats["entropy_per_particle_mean"] = []
        else:
            stats["entropy_mean"] = (
                float(entropy_np[active].mean()) if active.any() else 0.0
            )
            stats["entropy_std"] = (
                float(entropy_np[active].std()) if active.any() else 0.0
            )
            stats["entropy_max"] = (
                float(entropy_np[active].max()) if active.any() else 0.0
            )
            with np.errstate(invalid="ignore"):
                per_particle = np.where(active, entropy_np, np.nan)
                stats["entropy_per_particle_mean"] = [
                    float(v) for v in np.nanmean(per_particle, axis=1)
                ]
        stats["chosen_idx"] = int(chosen_idx)
        stats["_arrays"] = {
            "surprisal": surprisal_np,
            "done": done_np,
            "alpha": np.asarray(stats["alpha_history"], dtype=np.float32),
            "alpha_branching": np.asarray(
                stats["alpha_branching_history"], dtype=np.float32
            ),
            "alpha_entropy_mean": np.asarray(
                stats["alpha_entropy_mean_history"], dtype=np.float32
            ),
            "alpha_entropy_ema": np.asarray(
                stats["alpha_entropy_ema_history"], dtype=np.float32
            ),
            "alpha_entropy_ratio": np.asarray(
                stats["alpha_entropy_ratio_history"], dtype=np.float32
            ),
            "alpha_curve_power": np.asarray(
                stats["alpha_curve_power_history"], dtype=np.float32
            ),
            "alpha_progress": np.asarray(
                stats["alpha_progress_history"], dtype=np.float32
            ),
            "alpha_time_progress": np.asarray(
                stats["alpha_time_progress_history"], dtype=np.float32
            ),
            "alpha_cess_t": np.asarray(stats["alpha_cess_t_history"], dtype=np.int32),
            "alpha_cess_ratio": np.asarray(
                stats["alpha_cess_ratio_history"], dtype=np.float32
            ),
            "alpha_stage": np.asarray(stats["alpha_stage_history"], dtype=np.float32),
            "resample_branching": np.asarray(
                [
                    np.nan if v is None else v
                    for v in stats["resample_branching_history"]
                ],
                dtype=np.float32,
            ),
            "resample_ess_pred": np.asarray(
                [
                    np.nan if v is None else v
                    for v in stats["resample_ess_pred_history"]
                ],
                dtype=np.float32,
            ),
            "adaptive_ess_ratio": np.asarray(
                stats["adaptive_ess_ratio_history"], dtype=np.float32
            ),
            "proposal_temperature": np.asarray(
                stats["proposal_temperature_history"], dtype=np.float32
            ),
            "resample_mask": resample_mask[:t_gen].numpy(),
            "block_mask": block_mask[:t_gen].numpy(),
        }
        if entropy_np is not None:
            stats["_arrays"]["entropy"] = entropy_np
    if debug:
        print(
            f"{tag} ===== end   T_gen={cur - prompt_len} resample_count={stats['resample_count']} blocks={len(stats.get('block_history', []))} chosen_idx={chosen_idx} =====",
            flush=True,
        )
        if telemetry and stats["resample_t"]:
            print(f"{tag} resample @ t = {stats['resample_t']}", flush=True)
        if bool(cfg.debug_final_text):
            n_show = min(int(cfg.debug_particles), N)
            print(f"{tag} final particles (top {n_show} by weight):", flush=True)
            order = torch.argsort(w, descending=True)[:n_show].tolist()
            for i in order:
                mark = "<= chosen" if i == chosen_idx else ""
                h = (
                    f"{stats['entropy_per_particle_mean'][i]:.3f}"
                    if stats.get("entropy_per_particle_mean")
                    else "n/a"
                )
                print(
                    f"{tag}   p{i:<3} w={float(w[i]):.4f} logw={float(log_w[i]):9.2f} H(mean)={h} {mark}",
                    flush=True,
                )
                print(
                    f"{tag}        ...{_tail_text(tokenizer, seqs_out[i], prompt_len, cur, int(cfg.debug_text_chars) * 2)}",
                    flush=True,
                )
    return {
        "sequences": seqs_out,
        "log_w": log_w,
        "w": w,
        "log_z_partial": float(log_z_partial),
        "chosen_idx": chosen_idx,
        "chosen_sequence": chosen_sequence,
        "stats": stats,
    }


@torch.no_grad()
def easps_power_sample_multiround(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cfg: EASPSSamplingConfig,
    physical_batch: int = 16,
    n_rounds: int = 2,
) -> Dict[str, Any]:
    all_sequences = []
    all_log_w = []
    all_log_z = []
    all_stats = []
    round_cfg = EASPSSamplingConfig(
        **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    )
    round_cfg.n_particles = physical_batch
    for r in range(n_rounds):
        round_cfg.seed = int(cfg.seed) + r
        result = easps_power_sample_memopt(
            model, tokenizer, input_ids, round_cfg, round_idx=r
        )
        all_sequences.append(result["sequences"])
        all_log_w.append(result["log_w"])
        all_log_z.append(result.get("log_z_partial", 0.0))
        all_stats.append(result.get("stats", {}))
        del result
        torch.cuda.empty_cache()
    combined_log_w = _combine_round_log_w(
        all_log_w, all_log_z, getattr(cfg, "multiround_combine", "logz")
    )
    lw = combined_log_w - _logsumexp(combined_log_w, dim=0)
    w = torch.exp(lw)
    g = torch.Generator(device=input_ids.device)
    g.manual_seed(int(cfg.seed) + 10000)
    chosen_global = int(torch.multinomial(w, 1, generator=g).item())
    round_idx = chosen_global // physical_batch
    particle_idx = chosen_global % physical_batch
    chosen_sequence = all_sequences[round_idx][particle_idx]
    combined_stats = {
        "method": "EASPS",
        "multiround": True,
        "n_rounds": int(n_rounds),
        "physical_batch": int(physical_batch),
        "n_total_particles": int(n_rounds) * int(physical_batch),
        "ess_threshold": float(cfg.ess_threshold),
        "entropy_signal": "none",
        "block_size": int(cfg.block_size),
        "alpha_schedule": "exp",
        "alpha_bridge_schedule": "exp",
        "alpha_formula_version": EXP_FORMULA_VERSION,
        "proposal_from_alpha": bool(False),
        "alpha_branch_gamma": float(1.0),
        "resample_branch_lambda": float(0.5),
        "alpha_cess_target": float(0.8),
        "alpha_exp_k": float(cfg.alpha_exp_k),
        "alpha_entropy_ema_beta": float(0.9),
        "alpha_edt_lambda": float(1.0),
        "chosen_round": int(round_idx),
        "chosen_particle": int(particle_idx),
        "multiround_combine": str(getattr(cfg, "multiround_combine", "logz")),
        "log_z_partial_per_round": [float(z) for z in all_log_z],
        "round_weight_share": [
            float(w[r * physical_batch : (r + 1) * physical_batch].sum().item())
            for r in range(n_rounds)
        ],
        "round_stats": all_stats,
        "resample_count": sum((int(s.get("resample_count", 0)) for s in all_stats)),
        "alpha_branching_history_per_round": [
            s.get("alpha_branching_history") for s in all_stats
        ],
        "resample_branching_history_per_round": [
            s.get("resample_branching_history") for s in all_stats
        ],
        "alpha_cess_t_history_per_round": [
            s.get("alpha_cess_t_history") for s in all_stats
        ],
        "alpha_cess_ratio_history_per_round": [
            s.get("alpha_cess_ratio_history") for s in all_stats
        ],
        "alpha_stage_history_per_round": [
            s.get("alpha_stage_history") for s in all_stats
        ],
        "resample_ess_pred_history_per_round": [
            s.get("resample_ess_pred_history") for s in all_stats
        ],
        "adaptive_ess_ratio_history_per_round": [
            s.get("adaptive_ess_ratio_history") for s in all_stats
        ],
        "resample_mask_per_round": [s.get("resample_mask") for s in all_stats],
        "resample_t_per_round": [s.get("resample_t") for s in all_stats],
        "entropy_mean_per_round": [s.get("entropy_mean") for s in all_stats],
        "alpha_history_per_round": [s.get("alpha_history") for s in all_stats],
        "alpha_entropy_ema_history_per_round": [
            s.get("alpha_entropy_ema_history") for s in all_stats
        ],
        "final_ess_per_round": [s.get("final_ess") for s in all_stats],
        "entropy_per_particle_mean": [
            v for s in all_stats for v in s.get("entropy_per_particle_mean") or []
        ],
        "ess_history": [v for s in all_stats for v in s.get("ess_history", [])],
        "done_at_per_round": [s.get("done_at") for s in all_stats],
        "t_gen_per_round": [s.get("t_gen") for s in all_stats],
    }
    if bool(cfg.debug_verbose):
        print(
            f"{_DBG} ===== multiround done: rounds={n_rounds} x {physical_batch} total_resample={combined_stats['resample_count']} chosen=(round {round_idx}, particle {particle_idx}) =====",
            flush=True,
        )
    return {
        "chosen_sequence": chosen_sequence,
        "w": w,
        "chosen_global_idx": chosen_global,
        "round_idx": round_idx,
        "particle_idx": particle_idx,
        "n_total_particles": n_rounds * physical_batch,
        "stats": combined_stats,
    }
