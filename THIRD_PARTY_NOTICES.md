# Third Party Notices

## InferSim

This project derives analytical FLOP, KV-cache, communication concepts and
effective H20/H800/H200/PRO5000 profiles from Alibaba InferSim:

https://github.com/alibaba/InferSim

The full upstream CLI, layers, adapters, benchmark runners and kernel tables are
NOT included or executed. The upstream Apache 2.0 license text is not bundled in
this folder. Confirm and include the applicable upstream license and preserve
the relevant upstream notices before redistributing material derived from that
project.

Modified/reimplemented parts: `catalog.py` retains effective hardware values;
`performance_estimator.py` rewrites calculations with explicit SI units, closed-batch agg,
stage-capacity disagg, PP memory accounting and conservative sequential latency.
`tools.py`, `agent.py`, input contracts, reporting and search are project-owned.
Predictions do not inherit InferSim's measured-kernel calibration.

## Model Metadata

Users supply the exact Hugging Face `config.json` for their target model under
`src/data/`. No model weights or tokenizer files are bundled.
Weight and KV precision are explicit input assumptions, not a validation of an
actual engine's quantization support.

## AIConfigurator evaluation data

`evaluation/aiconfigurator_dense_silicon.csv` is a filtered and normalized
derivative subset of NVIDIA AIConfigurator's
`aic-core/src/aiconfigurator_core/systems/silicon_sample.csv`:

https://github.com/NVIDIA/AIConfigurator

The upstream AIConfigurator repository declares Apache License 2.0. The
normalized subset retains upstream measurement IDs and source metadata; the
importer, selection criteria, and any inferred BF16 precision are documented in
`evaluation/README.md` and `evaluation/import_aiconfigurator_silicon.py`. This
project does not include the complete upstream CSV. The source repository's
Apache-2.0 license applies to its covered material; consult the upstream license
for full terms:

https://www.apache.org/licenses/LICENSE-2.0
