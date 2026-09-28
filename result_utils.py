import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def smc_round_stats(stats):
    if not stats:
        return []
    return stats.get("round_stats") or [stats]


def smc_chosen_entropy(stats):
    rounds = smc_round_stats(stats)
    if not rounds:
        return {}, None
    r = int(stats.get("chosen_round", 0))
    p = int(stats.get("chosen_particle", stats.get("chosen_idx", 0)))
    if r >= len(rounds):
        return {}, None
    arrays = (rounds[r] or {}).get("_arrays")
    if not arrays or arrays.get("entropy") is None:
        return {}, None
    ent = np.asarray(arrays["entropy"])[p]
    active = np.asarray(arrays["done"])[p] == 0
    live = ent[active]
    if live.size == 0:
        live = ent
    return (
        {
            "smc_chosen_round": r,
            "smc_chosen_particle": p,
            "smc_chosen_len": int(live.size),
            "smc_chosen_entropy_mean": float(live.mean()),
            "smc_chosen_entropy_std": float(live.std()),
            "smc_chosen_entropy_min": float(live.min()),
            "smc_chosen_entropy_max": float(live.max()),
        },
        ent,
    )


def save_smc_telemetry(stats, problem_idx, telemetry_dir, rel_to):
    arrays = {}
    for r, rstats in enumerate(smc_round_stats(stats)):
        block = (rstats or {}).get("_arrays")
        if not block:
            continue
        for key, value in block.items():
            arrays[f"r{r}_{key}"] = value
    if not arrays:
        return None

    _, chosen_ent = smc_chosen_entropy(stats)
    if chosen_ent is not None:
        arrays["chosen_entropy"] = chosen_ent
    os.makedirs(telemetry_dir, exist_ok=True)
    path = os.path.join(telemetry_dir, f"idx{problem_idx:04d}.npz")
    np.savez_compressed(path, **arrays)
    return os.path.relpath(path, rel_to)


def easps_telemetry_columns(stats, npz_path):
    rounds = smc_round_stats(stats)
    if not rounds:
        return {}

    masks = [(r or {}).get("resample_mask") or "" for r in rounds]
    per_particle = [
        v for r in rounds for v in ((r or {}).get("entropy_per_particle_mean") or [])
    ]
    ent_means = [
        (r or {}).get("entropy_mean")
        for r in rounds
        if (r or {}).get("entropy_mean") is not None
    ]
    alpha_histories = [
        [float(v) for v in ((r or {}).get("alpha_history") or [])] for r in rounds
    ]
    chosen_round = int(stats.get("chosen_round", 0))
    alpha_branching_histories = [
        (r or {}).get("alpha_branching_history") or [] for r in rounds
    ]
    resample_branching_histories = [
        (r or {}).get("resample_branching_history") or [] for r in rounds
    ]
    alpha_cess_t_histories = [
        (r or {}).get("alpha_cess_t_history") or [] for r in rounds
    ]
    alpha_cess_ratio_histories = [
        (r or {}).get("alpha_cess_ratio_history") or [] for r in rounds
    ]
    alpha_stage_histories = [(r or {}).get("alpha_stage_history") or [] for r in rounds]
    resample_ess_pred_histories = [
        (r or {}).get("resample_ess_pred_history") or [] for r in rounds
    ]
    adaptive_ess_ratio_histories = [
        (r or {}).get("adaptive_ess_ratio_history") or [] for r in rounds
    ]
    chosen_alpha_history = (
        alpha_histories[chosen_round]
        if 0 <= chosen_round < len(alpha_histories)
        else []
    )
    first_round = rounds[0] or {}
    alpha_bridge_schedule = (
        stats.get("alpha_bridge_schedule")
        or first_round.get("alpha_bridge_schedule")
        or first_round.get("alpha_schedule")
    )
    alpha_formula_version = stats.get(
        "alpha_formula_version",
        first_round.get("alpha_formula_version"),
    )
    proposal_from_alpha = stats.get(
        "proposal_from_alpha",
        first_round.get("proposal_from_alpha"),
    )
    alpha_exp_k = stats.get(
        "alpha_exp_k",
        first_round.get("alpha_exp_k"),
    )
    alpha_entropy_ema_beta = stats.get(
        "alpha_entropy_ema_beta",
        first_round.get("alpha_entropy_ema_beta"),
    )
    chosen_cols, _ = smc_chosen_entropy(stats)
    return {
        **chosen_cols,
        "easps_alpha_branch_gamma": stats.get(
            "alpha_branch_gamma", first_round.get("alpha_branch_gamma")
        ),
        "easps_resample_branch_lambda": stats.get(
            "resample_branch_lambda", first_round.get("resample_branch_lambda")
        ),
        "easps_alpha_cess_target": stats.get(
            "alpha_cess_target", first_round.get("alpha_cess_target")
        ),
        "easps_n_particles": stats.get("n_total_particles") or stats.get("n_particles"),
        "easps_alpha_branching_history_per_round": json.dumps(
            alpha_branching_histories
        ),
        "easps_resample_branching_history_per_round": json.dumps(
            resample_branching_histories
        ),
        "easps_alpha_cess_t_history_per_round": json.dumps(alpha_cess_t_histories),
        "easps_alpha_cess_ratio_history_per_round": json.dumps(
            alpha_cess_ratio_histories
        ),
        "easps_alpha_stage_history_per_round": json.dumps(alpha_stage_histories),
        "easps_resample_ess_pred_history_per_round": json.dumps(
            resample_ess_pred_histories
        ),
        "easps_adaptive_ess_ratio_history_per_round": json.dumps(
            adaptive_ess_ratio_histories
        ),
        "easps_final_ess_per_round": json.dumps(
            [(r or {}).get("final_ess") for r in rounds]
        ),
        "easps_alpha_before_final_catchup_per_round": json.dumps(
            [(r or {}).get("alpha_before_final_catchup") for r in rounds]
        ),
        "easps_rounds": stats.get("n_rounds", 1),
        "easps_physical_batch": stats.get("physical_batch"),
        "easps_alpha_bridge_schedule": alpha_bridge_schedule,
        "easps_alpha_formula_version": alpha_formula_version,
        "easps_alpha_edt_lambda": stats.get(
            "alpha_edt_lambda", first_round.get("alpha_edt_lambda")
        ),
        "easps_proposal_from_alpha": proposal_from_alpha,
        "easps_alpha_exp_k": alpha_exp_k,
        "easps_alpha_entropy_ema_beta": alpha_entropy_ema_beta,
        "easps_alpha_history": json.dumps(chosen_alpha_history),
        "easps_alpha_history_per_round": json.dumps(alpha_histories),
        "easps_multiround_combine": stats.get("multiround_combine"),
        "easps_log_z_partial_per_round": json.dumps(
            [round(float(v), 4) for v in (stats.get("log_z_partial_per_round") or [])]
        ),
        "easps_round_weight_share": json.dumps(
            [round(float(v), 5) for v in (stats.get("round_weight_share") or [])]
        ),
        "easps_ess_threshold": stats.get("ess_threshold"),
        "easps_entropy_signal": stats.get("entropy_signal"),
        "easps_block_size": stats.get("block_size"),
        "easps_t_gen": json.dumps([(r or {}).get("t_gen") for r in rounds]),
        "easps_resample_count": stats.get("resample_count"),
        "easps_resample_mask": "|".join(masks),
        "easps_block_mask": "|".join((r or {}).get("block_mask") or "" for r in rounds),
        "easps_resample_t": json.dumps([(r or {}).get("resample_t") for r in rounds]),
        "easps_ess_history": json.dumps(
            [round(float(v), 4) for v in stats.get("ess_history", [])]
        ),
        "easps_entropy_mean": float(np.mean(ent_means)) if ent_means else None,
        "easps_entropy_per_particle_mean": json.dumps(
            [round(float(v), 5) for v in per_particle]
        ),
        "easps_entropy_npz": npz_path,
    }


def smc_telemetry_columns(stats, npz_path):
    rounds = smc_round_stats(stats)
    if not rounds:
        return {}

    masks = [(r or {}).get("resample_mask") or "" for r in rounds]
    per_particle = [
        v for r in rounds for v in ((r or {}).get("entropy_per_particle_mean") or [])
    ]
    ent_means = [
        (r or {}).get("entropy_mean")
        for r in rounds
        if (r or {}).get("entropy_mean") is not None
    ]
    chosen_cols, _ = smc_chosen_entropy(stats)
    return {
        **chosen_cols,
        "smc_n_particles": stats.get("n_total_particles") or stats.get("n_particles"),
        "smc_rounds": stats.get("n_rounds", 1),
        "smc_physical_batch": stats.get("physical_batch"),
        "smc_multiround_combine": stats.get("multiround_combine"),
        "smc_log_z_partial_per_round": json.dumps(
            [round(float(v), 4) for v in (stats.get("log_z_partial_per_round") or [])]
        ),
        "smc_round_weight_share": json.dumps(
            [round(float(v), 5) for v in (stats.get("round_weight_share") or [])]
        ),
        "smc_ess_threshold": stats.get("ess_threshold"),
        "smc_block_size": stats.get("block_size"),
        "smc_t_gen": json.dumps([(r or {}).get("t_gen") for r in rounds]),
        "smc_resample_count": stats.get("resample_count"),
        "smc_resample_mask": "|".join(masks),
        "smc_block_mask": "|".join((r or {}).get("block_mask") or "" for r in rounds),
        "smc_resample_t": json.dumps([(r or {}).get("resample_t") for r in rounds]),
        "smc_ess_history": json.dumps(
            [round(float(v), 4) for v in stats.get("ess_history", [])]
        ),
        "smc_entropy_mean": float(np.mean(ent_means)) if ent_means else None,
        "smc_entropy_per_particle_mean": json.dumps(
            [round(float(v), 5) for v in per_particle]
        ),
        "smc_entropy_npz": npz_path,
    }


def result_row_idx(row):
    try:
        return int(float(row["idx"]))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def infer_result_method(path):
    stem = path.stem
    marker = "_math_"
    if marker in stem:
        stem = stem.split(marker, 1)[1]
    for suffix in ("_seed", "_temp", "_tok"):
        if suffix in stem:
            stem = stem.split(suffix, 1)[0]
            break
    return stem


def result_method_label(row, path):
    value = row.get("method")
    if value is not None and str(value).strip():
        return str(value).strip()
    return infer_result_method(path)


def same_result_float(value, expected):
    if value in (None, ""):
        return True
    try:
        return math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def same_result_int(value, expected):
    if value in (None, ""):
        return True
    try:
        return int(float(value)) == expected
    except (TypeError, ValueError, OverflowError):
        return False


def result_row_matches_run(
    row,
    model,
    dataset,
    temperature,
    max_new_tokens,
    max_prompt_tokens=None,
):
    if row.get("model") not in (None, "", model):
        return False
    if row.get("dataset") not in (None, "", dataset):
        return False
    if not same_result_float(row.get("temperature"), temperature):
        return False
    if not same_result_int(row.get("max_new_tokens"), max_new_tokens):
        return False

    if max_prompt_tokens is not None and not same_result_int(
        row.get("max_prompt_tokens"), max_prompt_tokens
    ):
        return False
    return True


def prompt_token_length(tokenizer, prompt):
    return len(tokenizer.encode(prompt))


def prompt_length_selection(
    dataset,
    spec,
    dataset_opts,
    model,
    tokenizer,
    cot,
    max_prompt_tokens,
):
    lengths = {}
    selected = []
    excluded = []
    for idx in range(len(dataset)):
        item = spec.build_item(dataset[idx], idx, dataset_opts)
        prompt = spec.build_prompt(item, model, tokenizer, cot)
        length = prompt_token_length(tokenizer, prompt)
        lengths[idx] = length
        if length < max_prompt_tokens:
            selected.append(idx)
        else:
            excluded.append(idx)
    return selected, excluded, lengths


def load_result_file(path):
    try:
        if path.suffix == ".json":
            with path.open(encoding="utf-8") as f:
                rows = json.load(f)
            return rows if isinstance(rows, list) else []

        previous_limit = csv.field_size_limit()
        try:
            csv.field_size_limit(sys.maxsize)
            with path.open(newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        finally:
            csv.field_size_limit(previous_limit)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"[Auto resume] skip unreadable result {path}: {exc}", flush=True)
        return []


def auto_resume_start(
    result_dir,
    methods,
    model,
    dataset,
    temperature,
    max_new_tokens,
    scan_start,
    scan_end,
    redo_last=False,
    max_prompt_tokens=None,
):
    max_idx = {method: scan_start - 1 for method in methods}
    result_dir = Path(result_dir)

    if result_dir.exists():
        paths = sorted(result_dir.glob("*.csv")) + sorted(result_dir.glob("*.json"))
        for path in paths:
            for row in load_result_file(path):
                label = result_method_label(row, path)
                if label not in max_idx or not result_row_matches_run(
                    row,
                    model,
                    dataset,
                    temperature,
                    max_new_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                ):
                    continue
                idx = result_row_idx(row)
                if idx is not None and scan_start <= idx < scan_end:
                    max_idx[label] = max(max_idx[label], idx)

    min_completed = min(max_idx.values(), default=scan_start - 1)
    for method in methods:
        print(f"[Auto resume] {method}: last_idx={max_idx[method]}", flush=True)

    if min_completed >= scan_end - 1:
        resume_idx = scan_end
    elif min_completed < scan_start:
        resume_idx = scan_start
    elif redo_last:
        resume_idx = min_completed
    else:
        resume_idx = min_completed + 1

    print(f"[Auto resume] resume_idx={resume_idx}", flush=True)
    return resume_idx
