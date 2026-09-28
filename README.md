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

Supported datasets: `MATH` (MATH500), `Countdown`, `MedQA`, and `HumanEval`. Evaluation data is included in `data/`.

| Model option | Checkpoint |
| --- | --- |
| `qwen` | `Qwen/Qwen2.5-7B` |
| `phi` | `microsoft/Phi-3.5-mini-instruct` |

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
