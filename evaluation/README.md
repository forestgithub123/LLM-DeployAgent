# Evaluation against AIConfigurator silicon measurements

## What this evaluation can establish

The project previously planned a five-model throughput-ranking evaluation. The
available AIConfigurator dataset does not contain measured output tokens/s/GPU,
so that experiment cannot be honestly run with these data. Instead, this
evaluation compares the analytical simulator with the dataset's measured TTFT
and TPOT, and tests whether it orders deployment configurations sensibly within
the same model, GPU, backend/version, precision and input/output-length group.

The committed, filtered dataset is
[`aiconfigurator_dense_silicon.csv`](aiconfigurator_dense_silicon.csv): 778
non-WIP, Dense, aggregate-mode measurements for three base model sizes:
Llama 3.1 8B, Llama 3.1 70B and Qwen3 32B. The Llama 3.1 8B BF16 and FP8
checkpoints are separate model IDs, but not separate base model sizes. Eligible
rows are restricted to H100/H200 SXM and BF16/FP8. The original source contains
other models, systems and/or modes that this simulator does not currently
support.

## Reproduce

Run from the `LLM-DeployAgent` directory using the project's Python environment:

```bash
python evaluation/run_silicon_validation.py
```

The script runs deterministically and does not call an LLM or require model
weights. It writes:

- `outputs/evaluation/silicon_validation_rows.csv` — measured and predicted
  latency values and per-row errors.
- `outputs/evaluation/silicon_validation_summary.json` — aggregate and per-model
  error statistics, ranking scores and caveats.

To rebuild the normalized subset from the original AIConfigurator checkout:

```bash
python evaluation/import_aiconfigurator_silicon.py \
  --source /path/to/aiconfigurator-main/aic-core/src/aiconfigurator_core/systems/silicon_sample.csv
```

The importer records each original row ID and source metadata. Its precision
normalization is explicit: reported BF16/FP8 is preserved, while a blank
precision is treated as BF16 only when the model snapshot declares BF16; see
`precision_basis` in the normalized CSV.

## Metrics and current result

The simulator evaluates the measured parallel configuration (TP/PP/DP and
batch size) rather than searching for its own best configuration. This avoids
comparing an arbitrary measured setup against a different, optimized plan.
Measured request latency is derived as `TTFT + (OSL - 1) * TPOT` because the
source does not provide a separate request-latency field. Ranking is calculated
only inside matching workload groups; pairwise accuracy asks what fraction of
configuration pairs have the same predicted and measured latency order. Top-1
agreement checks whether the configuration predicted to have the lowest latency
also has the lowest measured latency in each group.

Latest bundled run (778/778 rows evaluated; 0 skipped):

| Metric | Result |
|---|---:|
| TTFT MAE | 5,956.04 ms |
| TTFT MAPE / median absolute percentage error | 685.9% / 449.2% |
| TPOT MAE | 12.62 ms |
| TPOT MAPE / median absolute percentage error | 47.8% / 49.2% |
| Within-group latency pairwise order agreement | 94.98% (5,714 of 6,016 pairs) |
| Lowest-latency configuration top-1 agreement | 81.97% (50 of 61 groups) |

Interpretation: these results show that the current estimator is poor at
predicting absolute TTFT and TPOT on this benchmark. The high within-group
ranking score is a narrower, relative result; it does not offset or hide the
absolute error. It suggests the estimator may still be useful for coarse
shortlisting of configurations under comparable conditions, but this requires
more independent workloads and prospective validation before relying on it for
real deployment decisions. In particular, configurations in the same sweep
are correlated observations, not independent test samples.

## Important assumptions and limitations

- This is an evaluation over three Dense base model sizes, not five different
  model sizes, and it does not measure model-throughput ranking.
- There are no measured output tokens/s/GPU values in the source subset. Do not
  convert predicted throughput into “real” throughput labels.
- Original rows do not specify KV-cache precision or shared-prefix length. The
  simulator assumes BF16 KV cache and prefix length 0.
- Rows cover multiple GPU types, backends/versions, precisions and workloads.
  Absolute error aggregates this heterogeneous sample; configuration ranking
  is restricted to exact matching workload groups.
- The source reports TTFT and TPOT, not request latency. The formula above
  assumes a generated sequence of `OSL` tokens with `OSL - 1` inter-token gaps.
- The benchmark's backend, kernels, batching, and serving overhead are not
  calibrated into this project's analytical model. Large absolute errors are
  therefore an important finding, not a formatting artifact.

The former five-model throughput-ranking script and template are retained for
future use only. That evaluation becomes executable only after obtaining
matched measured throughput data for five supported Dense models under the same
hardware, serving stack, precision, workload, and SLA.

## Source

The source is NVIDIA AIConfigurator's `aic-core/src/aiconfigurator_core/systems/silicon_sample.csv`
from the local `aiconfigurator-main` checkout. The repository is licensed under
Apache-2.0; attribution and provenance are recorded in
[`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md). This project bundles a
filtered derivative subset, not the full upstream file. Refer to the upstream
project for its license and current data provenance:

- [NVIDIA AIConfigurator repository](https://github.com/NVIDIA/AIConfigurator)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
