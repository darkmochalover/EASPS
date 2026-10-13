# Entropy Adaptive Sequence-Level Power Sampling (EASPS)

This study proposes Entropy Adaptive Sequence-Level Power Sampling (EASPS) to improve language-model reasoning without additional training. EASPS determines a common intermediate exponent for each prompt from the mean next-token entropy of unfinished particles, adapting the strength of sequence-level sharpening to uncertainty during generation.

## Method

At each decoding step, EASPS sets the intermediate exponent as

$$
\alpha_t = 1 + (\alpha_{\mathrm{final}} - 1)\exp(-kH_t), \qquad k>0.
$$

Here, $H_t$ is the mean Shannon entropy of the base model's next-token distributions over unfinished particles within an SMC round. The exponent is shared within that round. Higher entropy yields a smaller exponent; lower entropy brings it closer to $\alpha_{\mathrm{final}}$.

Particles are sampled with a fixed proposal temperature. Importance weights are updated at every token, including a correction for the existing prefix when the exponent changes. ESS is checked at block boundaries for systematic resampling. After generation, a final weight correction brings the target exponent to $\alpha_{\mathrm{final}}$.

## Setup

The reference environment uses Python 3.12, PyTorch 2.5.1, and Transformers 4.41.0. Phi requires FlashAttention 2 and CUDA. Run the following commands from the repository root in a Linux environment with the CUDA toolkit:

```bash
python -m pip install torch==2.5.1 packaging ninja setuptools wheel
python -m pip install --no-build-isolation -r requirements.txt
```

Supported datasets: `MATH` (MATH500), `Countdown`, `MedQA`, and `HumanEval`. See the Datasets and Data Preparation section for dataset sources and setup instructions.

| Model option | Checkpoint |
| --- | --- |
| `qwen` | `Qwen/Qwen2.5-7B` |
| `phi` | `microsoft/Phi-3.5-mini-instruct` |


## Datasets and Data Preparation

The evaluation benchmarks used in this study are publicly available.
Dataset files and raw experimental outputs are not included in this
repository. Download the benchmark datasets from the sources below
and prepare them in the `data/` directory before running experiments.

| Benchmark | Public source | Expected local file |
| --- | --- | --- |
| MATH500 | [OpenAI PRM800K](https://github.com/openai/prm800k), [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | `data/MATH500.json` |
| HumanEval | [OpenAI HumanEval](https://github.com/openai/human-eval) | `data/HumanEval.jsonl` |
| MedQA | [MedQA](https://github.com/jind11/MedQA) | `data/medQA/questions/US/4_options/phrases_no_exclude_test.jsonl` |
| Countdown | [Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4) | `data/countdown_cd3_test.jsonl` |

### Prepare the datasets

Run the following commands from the repository root.

Create the dataset directories:

```bash
mkdir -p data/medQA/questions/US/4_options
```

**MATH500 (500 problems)**

Download the public MATH-500 test split and convert it to the
`prompt`/`answer` JSON format expected by the evaluation code.

```bash
curl -fL \
  https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl \
  -o data/MATH500_source.jsonl

python - <<'PY'
import json
from pathlib import Path

src = Path("data/MATH500_source.jsonl")
rows = [
    json.loads(line)
    for line in src.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
assert len(rows) == 500

records = [
    {"prompt": row["problem"], "answer": row["answer"]}
    for row in rows
]
Path("data/MATH500.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
PY
```

**HumanEval (164 problems)**

Download the official compressed evaluation file and decompress it.

```bash
curl -fL \
  https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz \
  -o data/HumanEval.jsonl.gz

gzip -dc data/HumanEval.jsonl.gz > data/HumanEval.jsonl
```

**MedQA (USMLE, four options, test split)**

Obtain `phrases_no_exclude_test.jsonl` from the US four-option
data in the [official MedQA release](https://github.com/jind11/MedQA).

A publicly available copy can also be downloaded from
[Hugging Face](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).

```bash
curl -fL \
  https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options/resolve/main/phrases_no_exclude_test.jsonl \
  -o data/medQA/questions/US/4_options/phrases_no_exclude_test.jsonl
```

**Countdown (256 evaluation problems)**

The evaluation code expects a fixed 256-example, three-number
Countdown test set, stored as JSONL records with `input` and
`output` fields.

The original [Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)
provides the source benchmark. A public 256-example evaluation
snapshot in a compatible format is available in
[GDSD](https://github.com/GaryBall/GDSD/blob/main/dataset/countdown_cd3_test.jsonl).

To use that snapshot:

```bash
curl -fL \
  https://raw.githubusercontent.com/GaryBall/GDSD/main/dataset/countdown_cd3_test.jsonl \
  -o data/countdown_cd3_test.jsonl
```

**Note:** Exact reproduction of the reported Countdown results
requires the same 256 evaluation examples used in the study.
The public snapshot above should be checked against the original
experimental dataset before treating it as identical.

### Expected directory structure

```text
data/
├── MATH500.json
├── HumanEval.jsonl
├── countdown_cd3_test.jsonl
└── medQA/
    └── questions/
        └── US/
            └── 4_options/
                └── phrases_no_exclude_test.jsonl
```

The `data/` and `results/` directories are excluded from Git
tracking. Model generations, evaluation results, and telemetry
files are saved locally under `results/` during execution.


## Run

Run EASPS on MATH500 with Qwen2.5-7B:

```bash
bash run_math_qwen.sh
```

The example uses 16 particles in one round, proposal temperature $\tau=0.25$, $\alpha_{\mathrm{final}}=4$, $k=1$, a 64-token block, and ESS threshold $\kappa=0.5$. The runner sets $\alpha_{\mathrm{final}}=1/\tau$.

Configure runs through environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MODEL` / `DATASET` | `qwen` / `MATH` | Model and dataset |
| `TEMPERATURE` | `0.25` | Fixed proposal temperature |
| `ALPHA_EXP_K` | `1.0` | Entropy sensitivity |
| `PARTICLES` / `PHYSICAL_BATCH` | `16` / `16` | Particle budget and particles per round |
| `MAX_NEW_TOKENS` | `2048` | Generation limit |
| `LIMIT` | `-1` | Number of examples; `-1` uses all |

For a short run, use `LIMIT=10 bash run_math_qwen.sh`. Set `DRY_RUN=true` to inspect the configuration or `AUTO_RESUME=true` to resume saved results. Full options are available with `python power_samp.py --help`.

### Method selection

Set the corresponding variable to `true` to enable a method. The example script enables only EASPS by default.

| Method | Environment variable | Python option |
| --- | --- | --- |
| EASPS | `RUN_EASPS` | `--run_easps` |
| Base | `RUN_STD` | `--run_std` |
| Low-temperature | `RUN_NAIVE` | `--run_naive` |
| Power-SMC | `RUN_SMC` | `--run_smc` |
| MCMC | `RUN_MCMC` | `--run_mcmc` |

When calling `power_samp.py` directly, all five methods are enabled by default. Use `--run_<method> false` to disable individual methods.

## Outputs

The example script saves results under `results/<model>/<DATASET>/<run_tag>/`. CSV files contain generations, evaluation results, and sampling statistics; NPZ files store telemetry. Set `SAVE_FORMAT=json` for JSON output.

## License

See [LICENSE](LICENSE).
