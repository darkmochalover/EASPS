import math
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from easps_sampling import (
    _build_prompt_cache,
    _compact_cache_for_decode,
    _fmt_mask,
    _forward_prompt,
    _has_nonempty_boxed,
    _logical_logits_from_physical,
    _logsumexp,
    _resample_compacted_cache,
    _resolve_eos_token_ids,
    _uses_phi_dynamic_cache,
    apply_repetition_penalty_,
    effective_sample_size,
    expand_cache,
    reorder_past_key_values,
    select_cache_subset,
    systematic_resample,
    top_k_filter_,
    top_p_filter_,
    _DBG,
    _dbg_tag,
    _fmt_mult,
    _tail_text,
    _combine_round_log_w,
)


@dataclass
class PowerSMCSamplingConfig:
    max_new_tokens: int = 2048
    alpha: float = 4.0
    alpha_ramp_tokens: int = 100
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

    def __post_init__(self):
        for name in ("max_new_tokens", "n_particles", "block_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not math.isfinite(self.alpha) or self.alpha < 1.0:
            raise ValueError("alpha must be finite and >= 1.0")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and > 0")
        if (
            not math.isfinite(self.ess_threshold)
            or not 0.0 <= self.ess_threshold <= 1.0
        ):
            raise ValueError("ess_threshold must be in [0, 1]")


def _alpha_ramp(t, alpha_final, ramp_T):
    if ramp_T <= 1:
        return float(alpha_final)
    if t < ramp_T:
        frac = float(t + 1) / float(ramp_T)
        return 1.0 + (float(alpha_final) - 1.0) * frac
    return float(alpha_final)


@torch.no_grad()
def power_smc_sample_memopt(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cfg: PowerSMCSamplingConfig,
    round_idx: Optional[int] = None,
) -> Dict[str, Any]:
    assert input_ids.dim() == 2 and input_ids.size(0) == 1
    device = input_ids.device
    g = torch.Generator(device=device)
    g.manual_seed(int(cfg.seed))
    debug = bool(cfg.debug_verbose)
    telemetry = bool(cfg.collect_telemetry) or bool(cfg.debug_all) or debug
    tag = _dbg_tag(round_idx)
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
    stats = {
        "method": "Power-SMC",
        "resample_count": 0,
        "ess_history": [],
        "mean_logw_history": [],
        "max_logw_history": [],
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
        entropy_hist = torch.zeros((N, Tmax), dtype=torch.float32, device=device)
        surprisal_hist = torch.zeros((N, Tmax), dtype=torch.float32, device=device)
        done_hist = torch.zeros((N, Tmax), dtype=torch.uint8, device=device)
        resample_mask = torch.zeros(Tmax, dtype=torch.uint8)
        block_mask = torch.zeros(Tmax, dtype=torch.uint8)
        stats["resample_events"] = []
        stats["block_history"] = []
    if debug:
        print(
            f"{tag} ===== start  N={N} prompt_len={prompt_len} max_new={Tmax} block={int(cfg.block_size)} ess_thr={float(cfg.ess_threshold)}(*N={float(cfg.ess_threshold) * N:.1f}) alpha={alpha_final} ramp={ramp_T} temp={float(cfg.temperature)} =====",
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
        base_logprobs_phys = F.log_softmax(last_logits.float(), dim=-1)
        if cfg.temperature == 1.0:
            prop_logits_phys = last_logits.float()
        else:
            prop_logits_phys = last_logits.float() / cfg.temperature
        if active_cache_compaction:
            base_logprobs = _logical_logits_from_physical(
                base_logprobs_phys, logical_to_physical
            )
            prop_logits = _logical_logits_from_physical(
                prop_logits_phys, logical_to_physical
            )
        elif cow_active and logical_to_physical is not None:
            base_logprobs = base_logprobs_phys[logical_to_physical]
            prop_logits = prop_logits_phys[logical_to_physical]
        else:
            base_logprobs = base_logprobs_phys
            prop_logits = prop_logits_phys
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
            step_entropy = -(base_logprobs.exp() * base_logprobs).sum(dim=-1)
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
        alpha_t = _alpha_ramp(t=t, alpha_final=alpha_final, ramp_T=ramp_T)
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
                ent_t = entropy_hist[:, t]
                print(
                    f"{tag} {t:>5} {cur:>5} {int(done.sum()):>3}/{N:<3} {ent_t.mean():>8.3f} {ent_t.min():>7.3f} {ent_t.max():>7.3f} {surprisal_hist[:, t].mean():>7.3f} {log_w.mean():>9.2f} {alpha_t:>6.2f} {n_physical:>5}",
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
                print(
                    f"{tag}   p{i:<3}{flag} H={entropy_hist[i, t]:.3f} | ...{_tail_text(tokenizer, seqs[i], start_tok, cur, int(cfg.debug_text_chars))}",
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
            ess_thr = float(cfg.ess_threshold) * N
            will_resample = ess < ess_thr
            if telemetry:
                block_mask[t] = 1
                resample_mask[t] = 1 if will_resample else 0
                ent_t = entropy_hist[:, t]
                stats["block_history"].append(
                    {
                        "t": int(t),
                        "cur": int(cur),
                        "ess": float(ess),
                        "ess_threshold": float(ess_thr),
                        "resampled": bool(will_resample),
                        "done": int(done.sum().item()),
                        "entropy_mean": float(ent_t.mean().item()),
                        "logw_mean": float(log_w.mean().item()),
                    }
                )
            if debug:
                ent_t = entropy_hist[:, t]
                verdict = "RESAMPLE" if will_resample else "keep    "
                print(
                    f"{tag} {t:>5} {cur:>5} {int(done.sum()):>3}/{N:<3} {ent_t.mean():>8.3f} {ent_t.min():>7.3f} {ent_t.max():>7.3f} {surprisal_hist[:, t].mean():>7.3f} {log_w.mean():>9.2f} {alpha_t:>6.2f} {n_physical:>5}   | BLOCK ess={ess:6.2f} thr={ess_thr:6.2f} -> {verdict}",
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
    if prev_alpha != alpha_final:
        log_w = log_w + float(alpha_final - prev_alpha) * cum_logp
        prev_alpha = alpha_final
    seqs_out = seqs[:, :cur]
    lw = log_w - _logsumexp(log_w, dim=0)
    w = torch.exp(lw)
    chosen_idx = int(torch.multinomial(w, 1, generator=g).item())
    chosen_sequence = seqs_out[chosen_idx]
    entropy_np = None
    if telemetry:
        t_gen = cur - prompt_len
        entropy_np = entropy_hist[:, :t_gen].cpu().numpy()
        surprisal_np = surprisal_hist[:, :t_gen].cpu().numpy()
        done_np = done_hist[:, :t_gen].cpu().numpy()
        active = done_np == 0
        stats["n_particles"] = int(N)
        stats["t_gen"] = int(t_gen)
        stats["ess_threshold"] = float(cfg.ess_threshold)
        stats["block_size"] = int(cfg.block_size)
        stats["resample_mask"] = _fmt_mask(resample_mask[:t_gen])
        stats["block_mask"] = _fmt_mask(block_mask[:t_gen])
        stats["resample_t"] = torch.nonzero(resample_mask[:t_gen]).flatten().tolist()
        stats["entropy_mean"] = (
            float(entropy_np[active].mean()) if active.any() else 0.0
        )
        stats["entropy_std"] = float(entropy_np[active].std()) if active.any() else 0.0
        stats["entropy_max"] = float(entropy_np[active].max()) if active.any() else 0.0
        stats["surprisal_mean"] = (
            float(surprisal_np[active].mean()) if active.any() else 0.0
        )
        with np.errstate(invalid="ignore"):
            per_particle = np.where(active, entropy_np, np.nan)
            stats["entropy_per_particle_mean"] = [
                float(v) for v in np.nanmean(per_particle, axis=1)
            ]
        stats["chosen_idx"] = int(chosen_idx)
        stats["_arrays"] = {
            "entropy": entropy_np,
            "surprisal": surprisal_np,
            "done": done_np,
            "resample_mask": resample_mask[:t_gen].numpy(),
            "block_mask": block_mask[:t_gen].numpy(),
        }
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
                    if telemetry
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
def power_smc_sample_multiround(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cfg: PowerSMCSamplingConfig,
    physical_batch: int = 16,
    n_rounds: int = 2,
) -> Dict[str, Any]:
    all_sequences = []
    all_log_w = []
    all_log_z = []
    all_stats = []
    round_cfg = PowerSMCSamplingConfig(
        **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
    )
    round_cfg.n_particles = physical_batch
    for r in range(n_rounds):
        round_cfg.seed = int(cfg.seed) + r
        result = power_smc_sample_memopt(
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
        "method": "Power-SMC",
        "multiround": True,
        "n_rounds": int(n_rounds),
        "physical_batch": int(physical_batch),
        "n_total_particles": int(n_rounds) * int(physical_batch),
        "ess_threshold": float(cfg.ess_threshold),
        "block_size": int(cfg.block_size),
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
        "resample_mask_per_round": [s.get("resample_mask") for s in all_stats],
        "resample_t_per_round": [s.get("resample_t") for s in all_stats],
        "entropy_mean_per_round": [s.get("entropy_mean") for s in all_stats],
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
