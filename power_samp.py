import argparse
import json
import os
import time
from dataclasses import replace
import numpy
import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm
from autoregressive_sampling import (
    AutoregressiveSampler,
    mcmc_power_samp,
    set_generation_seed,
)
from benchmark_datasets import DATASET_SPECS, get_dataset_spec
from countdown_dataset import expression_metrics as countdown_expression_metrics
from easps_sampling import (
    EASPSSamplingConfig,
    easps_power_sample_memopt,
    easps_power_sample_multiround,
)
from model_utils import MODEL_MAP, model_input_device, model_load_kwargs_for_model
from result_utils import (
    auto_resume_start,
    easps_telemetry_columns,
    prompt_length_selection,
    prompt_token_length,
    result_row_idx,
    result_row_matches_run,
    save_smc_telemetry,
    smc_telemetry_columns,
)
from smc_sampling import PowerSMCSamplingConfig as SMCSamplingConfig
from smc_sampling import power_smc_sample_memopt as smc_power_sample_memopt
from smc_sampling import power_smc_sample_multiround as smc_power_sample_multiround


def trust_remote_code_for_model(model_name):
    return str(model_name).strip().lower() == "phi"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run power-sampling comparisons (MATH / Countdown / MedQA / HumanEval)."
    )
    parser.add_argument(
        "--save_str", action="store", type=str, default="results/", dest="save_str"
    )
    parser.add_argument(
        "--model", action="store", default="qwen", type=str, choices=["qwen", "phi"]
    )
    parser.add_argument(
        "--temperature", action="store", default=0.25, type=float, dest="temperature"
    )
    parser.add_argument(
        "--dataset",
        action="store",
        default="MATH",
        choices=["MATH", "COUNTDOWN", "MEDQA", "HUMANEVAL"],
        type=str.upper,
        help="MATH | Countdown | MedQA | HumanEval",
    )
    parser.add_argument("--dataset_path", action="store", type=str, default="")
    parser.add_argument("--he_timeout", action="store", type=float, default=3.0)
    parser.add_argument("--cot", action="store", type=bool, default=True)
    parser.add_argument("--mcmc_steps", action="store", type=int, default=10)
    parser.add_argument(
        "--device",
        action="store",
        type=str,
        dest="device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--device_map", action="store", type=str, default="cuda:0")
    parser.add_argument("--batch_idx", action="store", type=int, default=0)
    parser.add_argument("--max_new_tokens", action="store", type=int, default=3072)
    parser.add_argument(
        "--max_prompt_tokens",
        action="store",
        type=int,
        default=0,
        help="Exclusive cutoff on the final encoded model prompt. 0 disables filtering; a value of 900 keeps prompts with <900 tokens.",
    )
    parser.add_argument("--shard_size", action="store", type=int, default=100)
    parser.add_argument("--start_offset", action="store", type=int, default=0)
    parser.add_argument("--limit", action="store", type=int, default=-1)
    parser.add_argument("--auto_resume", action="store", type=str, default="false")
    parser.add_argument("--resume_redo_last", action="store", type=str, default="false")
    parser.add_argument("--resume_result_dir", action="store", type=str, default="")
    parser.add_argument(
        "--save_format",
        action="store",
        type=str,
        default="csv",
        choices=["csv", "json"],
    )
    parser.add_argument("--single_file", action="store", type=str, default="true")
    parser.add_argument("--save_filename", action="store", type=str, default="")
    parser.add_argument("--save_every", action="store", type=int, default=1)
    parser.add_argument("--print_completions", action="store", type=str, default="true")
    parser.add_argument(
        "--completion_print_chars", action="store", type=int, default=4000
    )
    parser.add_argument("--particles", action="store", type=int, default=64)
    parser.add_argument("--block_size", action="store", type=int, default=64)
    parser.add_argument("--physical_batch", action="store", type=int, default=16)
    parser.add_argument("--run_naive", action="store", type=str, default="true")
    parser.add_argument("--run_std", action="store", type=str, default="true")
    parser.add_argument("--run_smc", action="store", type=str, default="true")
    parser.add_argument("--run_mcmc", action="store", type=str, default="true")
    parser.add_argument("--run_easps", action="store", type=str, default="true")
    parser.add_argument("--debug_all", action="store", type=str, default="false")
    parser.add_argument("--debug_verbose", action="store", type=str, default="false")
    parser.add_argument("--debug_every", action="store", type=int, default=32)
    parser.add_argument("--debug_particles", action="store", type=int, default=4)
    parser.add_argument("--debug_text_every", action="store", type=int, default=0)
    parser.add_argument("--debug_text_chars", action="store", type=int, default=160)
    parser.add_argument(
        "--collect_telemetry", action="store", type=str, default="false"
    )
    parser.add_argument(
        "--save_telemetry_npz", action="store", type=str, default="false"
    )
    parser.add_argument("--ess_threshold", action="store", type=float, default=0.5)
    parser.add_argument(
        "--alpha_bridge_schedule",
        action="store",
        type=str,
        choices=["exp"],
        default="exp",
        help="EASPS target alpha schedule (exp only)",
    )
    parser.add_argument("--alpha_ramp_tokens", action="store", type=int, default=100)
    parser.add_argument(
        "--alpha_exp_k",
        action="store",
        type=float,
        default=1.0,
        help="Entropy decay coefficient k (>= 0)",
    )
    parser.add_argument(
        "--proposal_from_alpha",
        action="store",
        type=str,
        choices=["false"],
        default="false",
    )
    parser.add_argument(
        "--entropy_signal",
        action="store",
        type=str,
        choices=["none"],
        default="none",
        help="Fixed ESS threshold",
    )
    parser.add_argument("--rounds", action="store", type=int, default=0)
    parser.add_argument("--top_p", action="store", type=float, default=1.0)
    parser.add_argument("--seed", action="store", type=int, default=0)
    parser.add_argument(
        "--empty_cache_on_resample", action="store", type=str, default="false"
    )
    parser.add_argument(
        "--multiround_combine",
        action="store",
        type=str,
        default="logz",
        choices=["logz", "mixture", "legacy"],
    )
    parser.add_argument("--run_tag", action="store", type=str, default="")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    def as_bool(value):
        return str(value).strip().lower() in {"true", "1", "yes", "y", "on"}

    if args.physical_batch > 0:
        args.physical_batch = max(1, min(int(args.physical_batch), int(args.particles)))
        auto_rounds = -(-int(args.particles) // int(args.physical_batch))
        if int(args.rounds) <= 0 or int(args.rounds) * int(args.physical_batch) < int(
            args.particles
        ):
            args.rounds = auto_rounds
        print(
            f"[multiround] particles={args.particles} physical_batch={args.physical_batch} -> rounds={args.rounds} (total={args.rounds * args.physical_batch}) combine={args.multiround_combine}",
            flush=True,
        )
    else:
        args.rounds = 1
    model = args.model
    device = args.device
    dataset_name = str(args.dataset).strip().upper()
    spec = get_dataset_spec(dataset_name, args.dataset_path)
    spec = replace(spec, path=args.dataset_path or DATASET_SPECS[dataset_name].path)
    dataset_tag = spec.name
    dataset_opts = {
        "model": args.model,
        "he_timeout": args.he_timeout,
        "dataset_split": spec.split,
    }
    cot = args.cot
    temp = args.temperature
    mcmc_steps = args.mcmc_steps
    max_new_tokens = args.max_new_tokens
    if args.max_prompt_tokens < 0:
        parser.error("--max_prompt_tokens must be >= 0 (0 disables filtering)")
    max_prompt_tokens = (
        int(args.max_prompt_tokens) if int(args.max_prompt_tokens) > 0 else None
    )
    run_naive = as_bool(args.run_naive)
    run_std = as_bool(args.run_std)
    run_smc = as_bool(args.run_smc)
    run_mcmc = as_bool(args.run_mcmc)
    run_easps = as_bool(args.run_easps)
    print_completions = as_bool(args.print_completions)
    print_chars = args.completion_print_chars
    debug_verbose = as_bool(args.debug_verbose)
    debug_all = as_bool(args.debug_all) or debug_verbose
    collect_telemetry = as_bool(args.collect_telemetry) or debug_all
    save_npz = as_bool(args.save_telemetry_npz) and collect_telemetry
    entropy_signals = ["none"]

    def easps_label(signal):
        return f"easps_{signal}"

    easps_labels = [easps_label(s) for s in entropy_signals] if run_easps else []
    enabled_methods = [
        label
        for label, enabled in (
            ("naive", run_naive),
            ("std", run_std),
            ("smc", run_smc),
            ("mcmc", run_mcmc),
        )
        if enabled
    ] + easps_labels
    if not enabled_methods:
        print("[Run] nothing to run: no enabled methods")
        return 0
    run_tag = args.run_tag.strip()
    if run_tag:
        save_dir = os.path.join(args.save_str, model, dataset_name, run_tag)
    else:
        save_dir = os.path.join(args.save_str, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    single_file = as_bool(args.single_file)

    def method_tag(method):
        if method in easps_labels:
            tag = f"EASPS_alpha_bridge_schedule_exp_k_{args.alpha_exp_k:g}_formula_h_no_progress_v1_proposal_from_alpha_false"
        elif not run_tag:
            tag = method
        elif run_tag.lower() == method or run_tag.lower().startswith(f"{method}_"):
            tag = run_tag
        else:
            tag = f"{method}_{run_tag}"
        if not single_file:
            tag = f"{tag}_shard{args.batch_idx}"
        return tag

    method_tags = {method: method_tag(method) for method in enabled_methods}
    save_paths = {
        method: os.path.join(save_dir, f"{method_tags[method]}.{args.save_format}")
        for method in enabled_methods
    }
    if args.save_filename.strip():
        custom_name = os.path.basename(args.save_filename.strip())
        custom_stem, custom_ext = os.path.splitext(custom_name)
        custom_ext = custom_ext or f".{args.save_format}"
        custom_name = f"{custom_stem}{custom_ext}"
        if len(enabled_methods) == 1:
            save_paths[enabled_methods[0]] = os.path.join(save_dir, custom_name)
        else:
            save_paths = {
                method: os.path.join(save_dir, f"{custom_stem}_{method}{custom_ext}")
                for method in enabled_methods
            }
    telemetry_dirs = {
        method: os.path.join(
            save_dir,
            "telemetry",
            method_tags[method],
            model,
            f"temp_{temp}_tok_{max_new_tokens}",
        )
        for method in enabled_methods
    }
    model_str = MODEL_MAP[model]
    dataset = spec.load(spec.path)
    print(f"dataset done : {spec.name} ({len(dataset)} items) <- {spec.path}")
    shard_size = len(dataset) if args.shard_size <= 0 else args.shard_size
    shard_start = args.batch_idx * shard_size
    requested_start = shard_start + args.start_offset
    shard_end = min(shard_start + shard_size, len(dataset))
    run_indices = list(range(requested_start, shard_end))
    if as_bool(args.auto_resume) and enabled_methods:
        resume_result_dir = args.resume_result_dir.strip() or save_dir
        print(f"[Auto resume] result_dir={resume_result_dir}", flush=True)
        start = auto_resume_start(
            resume_result_dir,
            enabled_methods,
            model,
            dataset_tag,
            temp,
            max_new_tokens,
            requested_start,
            shard_end,
            redo_last=as_bool(args.resume_redo_last),
            max_prompt_tokens=max_prompt_tokens,
        )
        run_indices = list(range(start, shard_end))
    if args.limit > 0:
        run_indices = run_indices[: args.limit]
    run_index_set = set(run_indices)
    start = run_indices[0] if run_indices else shard_end
    end = run_indices[-1] + 1 if run_indices else shard_end
    span = f"[{start}, {end})" if run_indices else "none"
    print(
        f"[shard] batch_idx={args.batch_idx} pending={len(run_indices)} span={span} / {len(dataset)}"
    )
    for method, path in save_paths.items():
        print(f"[save ] {method}: {path}")
    if not run_indices:
        print("[Run] nothing to run: no pending indices")
        return 0
    trust_remote_code = trust_remote_code_for_model(model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_str, trust_remote_code=trust_remote_code
    )
    prompt_lengths = {}
    if max_prompt_tokens is not None:
        selected_indices, excluded_indices, prompt_lengths = prompt_length_selection(
            dataset, spec, dataset_opts, model, tokenizer, cot, max_prompt_tokens
        )
        eligible_in_range = [idx for idx in selected_indices if idx in run_index_set]
        excluded_with_lengths = [(idx, prompt_lengths[idx]) for idx in excluded_indices]
        print(
            f"[prompt-filter] keep final encoded prompt tokens < {max_prompt_tokens}: selected={len(selected_indices)}/{len(dataset)} excluded={excluded_with_lengths}",
            flush=True,
        )
        if not eligible_in_range:
            print(
                f"[Run] no eligible prompts among {len(run_indices)} pending indices under prompt_tokens<{max_prompt_tokens}",
                flush=True,
            )
            return 0
    effective_device_map = (
        "none" if str(device).strip().lower().startswith("cpu") else args.device_map
    )
    model_load_kwargs = model_load_kwargs_for_model(
        model, device_map=effective_device_map
    )
    if model == "phi":
        print(
            f"[attention] model=phi implementation={model_load_kwargs['attn_implementation']}",
            flush=True,
        )
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_str, **model_load_kwargs
    )
    if "device_map" not in model_load_kwargs:
        hf_model = hf_model.to(device)
    hf_model.eval()
    input_device = model_input_device(hf_model)
    cuda_enabled = (
        torch.cuda.is_available() and torch.device(input_device).type == "cuda"
    )
    dispatched_devices = sorted(
        {str(value) for value in getattr(hf_model, "hf_device_map", {}).values()}
    )
    print(
        f"[device] input={input_device} map={effective_device_map} dispatched={dispatched_devices or [str(input_device)]}",
        flush=True,
    )
    autoreg_sampler = AutoregressiveSampler(hf_model, tokenizer, input_device)
    print("loaded models")
    count = 0
    rows_by_method = {method: [] for method in enabled_methods}
    for method, save_path in save_paths.items():
        if not os.path.exists(save_path):
            continue
        if args.save_format == "json":
            with open(save_path, encoding="utf-8") as f:
                existing_rows = json.load(f)
        else:
            existing_rows = pd.read_csv(save_path).to_dict("records")
        before = len(existing_rows)
        rows_by_method[method] = [
            row
            for row in existing_rows
            if not (
                result_row_idx(row) is not None
                and result_row_idx(row) in run_index_set
                and result_row_matches_run(
                    row,
                    model,
                    dataset_tag,
                    temp,
                    max_new_tokens,
                    max_prompt_tokens=max_prompt_tokens,
                )
            )
        ]
        print(
            f"[resume] {method}: 기존 {before}행 중 {len(rows_by_method[method])}행 유지 (현재 설정의 pending {len(run_indices)}개 idx만 덮어씀)"
        )

    def flush_rows():
        for method, method_rows in rows_by_method.items():
            if not method_rows:
                continue
            ordered = sorted(
                method_rows,
                key=lambda row: (
                    str(row.get("model", "")),
                    str(row.get("temperature", "")),
                    str(row.get("max_new_tokens", "")),
                    result_row_idx(row) if result_row_idx(row) is not None else 1 << 30,
                ),
            )
            save_path = save_paths[method]
            if args.save_format == "json":
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(ordered, f, ensure_ascii=False, indent=1)
            else:
                pd.DataFrame(ordered).to_csv(save_path, index=False)

    naive_time = []
    std_time = []
    smc_time = []
    mcmc_time = []
    easps_time = {s: [] for s in entropy_signals}
    naive_acc = 0
    std_acc = 0
    smc_acc = 0
    mcmc_acc = 0
    easps_acc = {s: 0 for s in entropy_signals}
    naive_mem = []
    std_mem = []
    smc_mem = []
    mcmc_mem = []
    easps_mem = {s: [] for s in entropy_signals}
    for problem in tqdm(run_indices, desc=f"Benchmark on {spec.name}"):
        data = dataset[problem]
        item = spec.build_item(data, problem, dataset_opts)
        question = item["question"]
        answer = item["gold"]
        input_text = spec.build_prompt(item, model, tokenizer, cot)
        prompt_tokens = prompt_lengths.get(problem)
        if prompt_tokens is None:
            prompt_tokens = prompt_token_length(tokenizer, input_text)
        if max_prompt_tokens is not None and prompt_tokens >= max_prompt_tokens:
            print(
                f"[prompt-filter] skip original idx={problem}: prompt_tokens={prompt_tokens} >= {max_prompt_tokens}",
                flush=True,
            )
            continue
        count += 1
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(input_device)
        prefx = [idx.item() for idx in input_ids[0]]
        naive_answer = std_answer = smc_answer = mcmc_answer = None
        naive_completion = std_completion = smc_completion = mcmc_completion = ""
        smc_extra = {}
        easps_answer_by_sig = {s: None for s in entropy_signals}
        easps_completion_by_sig = {s: "" for s in entropy_signals}
        easps_extra_by_sig = {s: {} for s in entropy_signals}
        if run_naive:
            set_generation_seed(args.seed + problem)
            t1 = time.time()
            if cuda_enabled:
                torch.cuda.reset_peak_memory_stats()
            if cuda_enabled:
                torch.cuda.synchronize()
            naive_temp_output = hf_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=False,
                do_sample=True,
                temperature=temp,
                repetition_penalty=1.0,
            )
            if cuda_enabled:
                torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3 if cuda_enabled else 0.0
            naive_mem.append(peak)
            t2 = time.time()
            naive_time.append(t2 - t1)
        if run_std:
            set_generation_seed(args.seed + problem)
            t1 = time.time()
            if cuda_enabled:
                torch.cuda.reset_peak_memory_stats()
            if cuda_enabled:
                torch.cuda.synchronize()
            std_output = hf_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=False,
                do_sample=True,
                repetition_penalty=1.0,
            )
            if cuda_enabled:
                torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3 if cuda_enabled else 0.0
            std_mem.append(peak)
            t2 = time.time()
            std_time.append(t2 - t1)
        alpha = 1.0 / temp
        for easps_signal in entropy_signals if run_easps else ():
            t1 = time.time()
            cfg = EASPSSamplingConfig(
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                n_particles=args.particles,
                ess_threshold=args.ess_threshold,
                temperature=temp,
                block_size=args.block_size,
                alpha_ramp_tokens=args.alpha_ramp_tokens,
                alpha_exp_k=args.alpha_exp_k,
                top_p=args.top_p,
                seed=args.seed,
                empty_cache_on_resample=as_bool(args.empty_cache_on_resample),
                multiround_combine=args.multiround_combine,
                stop_on_boxed=spec.stop_on_boxed,
                boxed_check_window_tokens=256,
                collect_telemetry=collect_telemetry,
                debug_all=debug_all,
                debug_verbose=debug_verbose,
                debug_every=args.debug_every,
                debug_particles=args.debug_particles,
                debug_text_every=args.debug_text_every,
                debug_text_chars=args.debug_text_chars,
            )
            if cuda_enabled:
                torch.cuda.reset_peak_memory_stats()
            if cuda_enabled:
                torch.cuda.synchronize()
            if args.physical_batch > 0:
                easps_out = easps_power_sample_multiround(
                    hf_model,
                    tokenizer,
                    input_ids,
                    cfg,
                    physical_batch=args.physical_batch,
                    n_rounds=args.rounds,
                )
            else:
                easps_out = easps_power_sample_memopt(
                    hf_model, tokenizer, input_ids, cfg
                )
            if cuda_enabled:
                torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3 if cuda_enabled else 0.0
            easps_mem[easps_signal].append(peak)
            t2 = time.time()
            easps_time[easps_signal].append(t2 - t1)
            easps_ids = easps_out["chosen_sequence"][len(input_ids[0]) :].to("cpu")
            easps_completion = tokenizer.decode(easps_ids, skip_special_tokens=True)
            easps_completion_by_sig[easps_signal] = easps_completion
            easps_stats = easps_out.get("stats") or {}
            npz_path = (
                save_smc_telemetry(
                    easps_stats,
                    problem,
                    telemetry_dirs[easps_label(easps_signal)],
                    save_dir,
                )
                if save_npz
                else None
            )
            easps_extra = easps_telemetry_columns(easps_stats, npz_path)
            easps_extra_by_sig[easps_signal] = easps_extra
            if easps_extra.get("smc_chosen_entropy_mean") is None:
                h_part = "H(chosen)=n/a (entropy not collected)"
            else:
                h_part = f"H(chosen) mean={easps_extra['smc_chosen_entropy_mean']:.4f} std={easps_extra['smc_chosen_entropy_std']:.4f} min={easps_extra['smc_chosen_entropy_min']:.4f} max={easps_extra['smc_chosen_entropy_max']:.4f} len={easps_extra['smc_chosen_len']}"
            print(
                f"[easps idx={problem} signal={easps_signal}] resample={easps_extra.get('easps_resample_count')}  {h_part}",
                flush=True,
            )
            del easps_out, easps_stats
            torch.cuda.empty_cache()
        if run_smc:
            t1 = time.time()
            cfg = SMCSamplingConfig(
                alpha=alpha,
                max_new_tokens=max_new_tokens,
                n_particles=args.particles,
                ess_threshold=args.ess_threshold,
                temperature=temp,
                block_size=args.block_size,
                alpha_ramp_tokens=100,
                top_p=args.top_p,
                seed=args.seed,
                empty_cache_on_resample=as_bool(args.empty_cache_on_resample),
                multiround_combine=args.multiround_combine,
                stop_on_boxed=spec.stop_on_boxed,
                boxed_check_window_tokens=256,
                collect_telemetry=collect_telemetry,
                debug_all=debug_all,
                debug_verbose=debug_verbose,
                debug_every=args.debug_every,
                debug_particles=args.debug_particles,
                debug_text_every=args.debug_text_every,
                debug_text_chars=args.debug_text_chars,
            )
            if cuda_enabled:
                torch.cuda.reset_peak_memory_stats()
            if cuda_enabled:
                torch.cuda.synchronize()
            if args.physical_batch > 0:
                smc_out = smc_power_sample_multiround(
                    hf_model,
                    tokenizer,
                    input_ids,
                    cfg,
                    physical_batch=args.physical_batch,
                    n_rounds=args.rounds,
                )
            else:
                smc_out = smc_power_sample_memopt(hf_model, tokenizer, input_ids, cfg)
            if cuda_enabled:
                torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3 if cuda_enabled else 0.0
            smc_mem.append(peak)
            t2 = time.time()
            smc_time.append(t2 - t1)
            smc_ids = smc_out["chosen_sequence"][len(input_ids[0]) :].to("cpu")
            smc_completion = tokenizer.decode(smc_ids, skip_special_tokens=True)
            smc_stats = smc_out.get("stats") or {}
            npz_path = (
                save_smc_telemetry(smc_stats, problem, telemetry_dirs["smc"], save_dir)
                if save_npz
                else None
            )
            smc_extra = smc_telemetry_columns(smc_stats, npz_path)
            if smc_extra.get("smc_chosen_entropy_mean") is None:
                h_part = "H(chosen)=n/a (telemetry off)"
            else:
                h_part = f"H(chosen) mean={smc_extra['smc_chosen_entropy_mean']:.4f} std={smc_extra['smc_chosen_entropy_std']:.4f} min={smc_extra['smc_chosen_entropy_min']:.4f} max={smc_extra['smc_chosen_entropy_max']:.4f} len={smc_extra['smc_chosen_len']}"
            print(
                f"[smc idx={problem}] resample={smc_extra.get('smc_resample_count')}  {h_part}",
                flush=True,
            )
            del smc_out, smc_stats
            torch.cuda.empty_cache()
        if run_mcmc:
            set_generation_seed(args.seed + problem)
            t1 = time.time()
            if cuda_enabled:
                torch.cuda.reset_peak_memory_stats()
            if cuda_enabled:
                torch.cuda.synchronize()
            mcmc_power_samp_output, _, _, acceptance_ratio = mcmc_power_samp(
                autoreg_sampler, prefx, temp, mcmc_steps, max_new_tokens=max_new_tokens
            )
            if cuda_enabled:
                torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1024**3 if cuda_enabled else 0.0
            mcmc_mem.append(peak)
            t2 = time.time()
            mcmc_time.append(t2 - t1)
            mcmc_power_samp_ids = torch.tensor(
                mcmc_power_samp_output[len(input_ids[0]) :],
                dtype=torch.long,
                device=input_device,
            ).to("cpu")
            mcmc_completion = tokenizer.decode(
                mcmc_power_samp_ids, skip_special_tokens=True
            )
        if run_naive:
            naive_generated_ids = (
                naive_temp_output[0][:, len(input_ids[0]) :].squeeze().to("cpu")
            )
            naive_completion = tokenizer.decode(
                naive_generated_ids, skip_special_tokens=True
            )
            naive_answer = spec.parse(naive_completion, item, dataset_opts)
        if run_std:
            std_generated_ids = (
                std_output[0][:, len(input_ids[0]) :].squeeze().to("cpu")
            )
            std_completion = tokenizer.decode(
                std_generated_ids, skip_special_tokens=True
            )
            std_answer = spec.parse(std_completion, item, dataset_opts)
        if run_smc:
            smc_answer = spec.parse(smc_completion, item, dataset_opts)
        if run_easps:
            for sig in entropy_signals:
                easps_answer_by_sig[sig] = spec.parse(
                    easps_completion_by_sig[sig], item, dataset_opts
                )
        if run_mcmc:
            mcmc_answer = spec.parse(mcmc_completion, item, dataset_opts)
        correct = {}
        method_results = [
            ("naive", run_naive, naive_answer, naive_completion, naive_time, naive_mem),
            ("std", run_std, std_answer, std_completion, std_time, std_mem),
            ("smc", run_smc, smc_answer, smc_completion, smc_time, smc_mem),
            ("mcmc", run_mcmc, mcmc_answer, mcmc_completion, mcmc_time, mcmc_mem),
        ]
        method_results += [
            (
                easps_label(sig),
                run_easps,
                easps_answer_by_sig[sig],
                easps_completion_by_sig[sig],
                easps_time[sig],
                easps_mem[sig],
            )
            for sig in entropy_signals
        ]
        for label, enabled, ans, comp, times, mems in method_results:
            if not enabled:
                continue
            correct[label] = int(bool(spec.grade(ans, item, comp, dataset_opts)))
            row = {
                "idx": problem,
                "method": label,
                "model": model,
                "dataset": dataset_tag,
                "dataset_source": spec.source or spec.path,
                "dataset_config": spec.config,
                "dataset_revision": spec.revision,
                "dataset_split": spec.split or item.get("meta", {}).get("split"),
                "dataset_path": spec.path,
                "dataset_subject": item.get("meta", {}).get("subject"),
                "dataset_item_id": item.get("meta", {}).get("dataset_id"),
                "dataset_source_idx": item.get("meta", {}).get("source_idx"),
                "dataset_item_url": item.get("meta", {}).get("source_url"),
                "temperature": temp,
                "seed": args.seed,
                "method_seed": args.seed + problem
                if label in {"naive", "std", "mcmc"}
                else args.seed,
                "max_new_tokens": max_new_tokens,
                "prompt_tokens": prompt_tokens,
                "max_prompt_tokens": max_prompt_tokens,
                "answer": ans,
                "gold": answer,
                "correct": correct[label],
                "time_sec": times[-1] if times else None,
                "peak_mem_gb": mems[-1] if mems else None,
                "completion": comp,
            }
            if dataset_name == "COUNTDOWN":
                countdown_metrics = countdown_expression_metrics(ans, item)
                row.update(
                    {
                        "countdown_numbers": ",".join(
                            (str(number) for number in countdown_metrics["numbers"])
                        ),
                        "countdown_target": countdown_metrics["target"],
                        "countdown_expression": countdown_metrics["expression"],
                        "countdown_value": countdown_metrics["value"],
                        "countdown_valid_syntax": int(
                            countdown_metrics["valid_syntax"]
                        ),
                        "countdown_uses_all_numbers": int(
                            countdown_metrics["uses_all_numbers"]
                        ),
                    }
                )
            if label == "smc":
                row.update(smc_extra)
            elif label in easps_labels:
                row.update(easps_extra_by_sig[label[len("easps_") :]])
            rows_by_method[label].append(row)
        naive_acc += correct.get("naive", 0)
        std_acc += correct.get("std", 0)
        smc_acc += correct.get("smc", 0)
        mcmc_acc += correct.get("mcmc", 0)
        for sig in entropy_signals:
            easps_acc[sig] += correct.get(easps_label(sig), 0)
        if count % max(args.save_every, 1) == 0:
            flush_rows()
        print("Question: ", question)
        print("Gold answer:", answer)
        if run_naive:
            print("Naive Acc: ", naive_acc / count)
            print("Naive time: ", numpy.mean(np.array(naive_time)))
            print("Naive answer:", naive_answer)
            print("Naive mem: ", numpy.mean(np.array(naive_mem)))
        if run_std:
            print("STD Acc: ", std_acc / count)
            print("STD time: ", numpy.mean(np.array(std_time)))
            print("STD answer: ", std_answer)
        if run_smc:
            print("SMC Acc: ", smc_acc / count)
            print("SMC time: ", numpy.mean(np.array(smc_time)))
            print("SMC answer:", smc_answer)
            print("SMC mem: ", numpy.mean(np.array(smc_mem)))
        if run_mcmc:
            print("MCMC Acc: ", mcmc_acc / count)
            print("MCMC time: ", numpy.mean(np.array(mcmc_time)))
            print("MCMC answer:", mcmc_answer)
            print("MCMC mem: ", numpy.mean(np.array(mcmc_mem)))
        if run_easps:
            for sig in entropy_signals:
                print(f"EASPS[{sig}] Acc: ", easps_acc[sig] / count)
                print(f"EASPS[{sig}] time: ", numpy.mean(np.array(easps_time[sig])))
                print(f"EASPS[{sig}] answer:", easps_answer_by_sig[sig])
                print(f"EASPS[{sig}] mem: ", numpy.mean(np.array(easps_mem[sig])))
        if print_completions:
            completions = [
                ("naive", run_naive, naive_completion),
                ("std", run_std, std_completion),
                ("smc", run_smc, smc_completion),
                ("mcmc", run_mcmc, mcmc_completion),
            ] + [
                (easps_label(sig), run_easps, easps_completion_by_sig[sig])
                for sig in entropy_signals
            ]
            for label, enabled, comp in completions:
                if enabled:
                    print(f"--- {label} completion ---\n{comp[:print_chars]}")
        print(
            "***************************************************************************************************"
        )
    flush_rows()
    for method, path in save_paths.items():
        print(f"[done] {method}: {len(rows_by_method[method])} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
