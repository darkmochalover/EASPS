#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export TQDM_DISABLE=${TQDM_DISABLE:-1}

MODEL=${MODEL:-qwen}
DATASET=${DATASET:-MATH}
TEMPERATURE=${TEMPERATURE:-0.25}
MCMC_STEPS=${MCMC_STEPS:-10}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}

START_OFFSET=${START_OFFSET:-0}
LIMIT=${LIMIT:-"-1"}

case "${MODEL}" in
  qwen|phi) ;;
  *) echo "[error] MODEL must be qwen or phi" >&2; exit 2 ;;
esac

case "${DATASET^^}" in
  MATH)      DEFAULT_DATASET_PATH=data/MATH500.json ;;
  COUNTDOWN) DEFAULT_DATASET_PATH=data/countdown_cd3_test.jsonl ;;
  MEDQA)     DEFAULT_DATASET_PATH=data/medQA/questions/US/4_options/phrases_no_exclude_test.jsonl ;;
  HUMANEVAL) DEFAULT_DATASET_PATH=data/HumanEval.jsonl ;;
  *) echo "[error] DATASET must be MATH, Countdown, MedQA, or HumanEval" >&2; exit 2 ;;
esac

DATASET_PATH=${DATASET_PATH:-${DEFAULT_DATASET_PATH}}

echo "[dataset] ${DATASET} "

SAVE_STR=${SAVE_STR:-results}
SAVE_FORMAT=${SAVE_FORMAT:-csv}
SAVE_EVERY=${SAVE_EVERY:-1}

PRINT_COMPLETIONS=${PRINT_COMPLETIONS:-false}
COMPLETION_PRINT_CHARS=${COMPLETION_PRINT_CHARS:-1000}

PARTICLES=${PARTICLES:-16}
PHYSICAL_BATCH=${PHYSICAL_BATCH:-16}
ROUNDS=${ROUNDS:-0}
BLOCK_SIZE=${BLOCK_SIZE:-64}

ALPHA_BRIDGE_SCHEDULE=exp
ALPHA_RAMP_TOKENS=${ALPHA_RAMP_TOKENS:-100}
PROPOSAL_FROM_ALPHA=false
ALPHA_EXP_K=${ALPHA_EXP_K:-1.0}
EXP_FORMULA_VERSION=h_no_progress_v1

TOP_P=${TOP_P:-1.0}
SEED=${SEED:-0}

ESS_THRESHOLD=${ESS_THRESHOLD:-0.5}

ENTROPY_SIGNAL=none

DEBUG_ALL=${DEBUG_ALL:-true}
DEBUG_VERBOSE=${DEBUG_VERBOSE:-false}
DEBUG_EVERY=${DEBUG_EVERY:-32}
DEBUG_PARTICLES=${DEBUG_PARTICLES:-4}
DEBUG_TEXT_EVERY=${DEBUG_TEXT_EVERY:-0}
DEBUG_TEXT_CHARS=${DEBUG_TEXT_CHARS:-160}
SAVE_TELEMETRY_NPZ=${SAVE_TELEMETRY_NPZ:-true}

RUN_NAIVE=${RUN_NAIVE:-false}
RUN_STD=${RUN_STD:-false}
RUN_SMC=${RUN_SMC:-false}
RUN_MCMC=${RUN_MCMC:-false}
RUN_EASPS=${RUN_EASPS:-true}

AUTO_RESUME=${AUTO_RESUME:-false}
RESUME_REDO_LAST=${RESUME_REDO_LAST:-false}
RESUME_RESULT_DIR_OVERRIDE=${RESUME_RESULT_DIR:-}
BASE_START_OFFSET=${START_OFFSET}

for alpha_exp_k in ${ALPHA_EXP_K}; do
  START_OFFSET=${BASE_START_OFFSET}

  RUN_TAG="EASPS_exp_no_progress_v1_k_${alpha_exp_k}_proposal_from_alpha_${PROPOSAL_FROM_ALPHA}"
  LC_NUMERIC=C printf -v alpha_exp_k_file '%g' "${alpha_exp_k}"
  EASPS_FILE_TAG="EASPS_alpha_bridge_schedule_exp_k_${alpha_exp_k_file}_formula_${EXP_FORMULA_VERSION}_proposal_from_alpha_${PROPOSAL_FROM_ALPHA}"
  RESUME_RESULT_DIR="${RESUME_RESULT_DIR_OVERRIDE:-${SAVE_STR}/${MODEL}/${DATASET^^}/${RUN_TAG}}"
  echo "########## alpha_exp_k=${alpha_exp_k} ess_threshold=${ESS_THRESHOLD} signals=[${ENTROPY_SIGNAL}] -> ${RESUME_RESULT_DIR} ##########"

  echo "[Run] requested_start=${START_OFFSET}, limit=${LIMIT} (dataset size is resolved in power_samp.py)"
  echo "[config] particles=${PARTICLES} physical_batch=${PHYSICAL_BATCH} rounds=${ROUNDS}(0=auto) block=${BLOCK_SIZE} ess_thr=${ESS_THRESHOLD} signals=[${ENTROPY_SIGNAL}] max_new=${MAX_NEW_TOKENS} temp=${TEMPERATURE} top_p=${TOP_P} seed=${SEED}"
  echo "[config] out=${SAVE_STR}/${MODEL}/${DATASET^^}/${RUN_TAG}/${EASPS_FILE_TAG}.${SAVE_FORMAT}"

  case "${DRY_RUN:-false}" in
    true|1|yes|y|on)
      echo "[dry-run] dataset=${DATASET^^} schedule=exp formula=${EXP_FORMULA_VERSION} k=${alpha_exp_k} run_tag=${RUN_TAG}"
      continue
      ;;
  esac

  "${PYTHON:-python}" -u power_samp.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --dataset_path "${DATASET_PATH}" \
    --single_file true \
    --temperature "${TEMPERATURE}" \
    --mcmc_steps "${MCMC_STEPS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --batch_idx 0 \
    --shard_size -1 \
    --start_offset "${START_OFFSET}" \
    --limit "${LIMIT}" \
    --auto_resume "${AUTO_RESUME}" \
    --resume_redo_last "${RESUME_REDO_LAST}" \
    --alpha_bridge_schedule "${ALPHA_BRIDGE_SCHEDULE}" \
    --alpha_ramp_tokens "${ALPHA_RAMP_TOKENS}" \
    --alpha_exp_k "${alpha_exp_k}" \
    --proposal_from_alpha "${PROPOSAL_FROM_ALPHA}" \
    --resume_result_dir "${RESUME_RESULT_DIR}" \
    --save_str "${SAVE_STR}" \
    --save_format "${SAVE_FORMAT}" \
    --save_every "${SAVE_EVERY}" \
    --print_completions "${PRINT_COMPLETIONS}" \
    --completion_print_chars "${COMPLETION_PRINT_CHARS}" \
    --particles "${PARTICLES}" \
    --physical_batch "${PHYSICAL_BATCH}" \
    --rounds "${ROUNDS}" \
    --block_size "${BLOCK_SIZE}" \
    --ess_threshold "${ESS_THRESHOLD}" \
    --entropy_signal "${ENTROPY_SIGNAL}" \
    --top_p "${TOP_P}" \
    --seed "${SEED}" \
    --run_tag "${RUN_TAG}" \
    --collect_telemetry true \
    --debug_all "${DEBUG_ALL}" \
    --debug_verbose "${DEBUG_VERBOSE}" \
    --debug_every "${DEBUG_EVERY}" \
    --debug_particles "${DEBUG_PARTICLES}" \
    --debug_text_every "${DEBUG_TEXT_EVERY}" \
    --debug_text_chars "${DEBUG_TEXT_CHARS}" \
    --save_telemetry_npz "${SAVE_TELEMETRY_NPZ}" \
    --run_naive "${RUN_NAIVE}" \
    --run_std "${RUN_STD}" \
    --run_smc "${RUN_SMC}" \
    --run_mcmc "${RUN_MCMC}" \
    --run_easps "${RUN_EASPS}" \
    "$@"
done
