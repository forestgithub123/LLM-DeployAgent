# LLM-DeployAgent: An Agentic Planner for LLM Inference Deployment

## Abstract

Choosing an inference deployment for a large language model is a constrained systems problem: model size, GPU memory, parallelism, batching, prompt length and latency targets interact, while exhaustive benchmarking is costly. LLM-DeployAgent turns a model configuration, GPU budget and service-level objectives into a shortlist of candidate deployments. A language model can iteratively inspect tool results and refine its search, while a deterministic analytical simulator—not the LLM—calculates performance estimates and verifies the final recommendation. This separation makes the system more inspectable than free-form advice. Evaluation against 778 NVIDIA AIConfigurator silicon measurements shows a clear limitation: absolute latency estimates are not accurate enough for deployment guarantees. However, within matched workload groups, the estimator orders configurations more consistently. The result is best understood as a prototype for early-stage screening, not a replacement for serving benchmarks.

## Problem, user and intended value

The primary persona is an ML infrastructure engineer or researcher who knows the target model and available GPU pool, but needs help narrowing down tensor/pipeline parallelism, batching, and whether to use aggregated (prefill and decode together) or disaggregated serving. Today, answering this often requires expertise, a cluster, and repeated backend-specific experiments. This project aims to make the first planning step more accessible: expose assumptions, identify plausible candidates, and explain bottlenecks before a costly deployment test.

The intended difference is not that an LLM predicts hardware performance by intuition. Users can state constraints, inspect the agent's trace, and receive a reproducible shortlist. A deterministic workflow can enumerate a fixed search space, but does not naturally redirect its search based on intermediate bottleneck evidence. In live mode the LLM can select tools and refine regions after seeing observations. Its contribution is adaptive orchestration and explanation; numeric estimates remain tool-owned. Scripted mode is a transparent no-API baseline.

## Product architecture and workflow

```text
User / infrastructure engineer
        | model config, GPU budget, workload, TTFT/TPOT targets
        v
     config.toml ----> Input validation and model/GPU profile loading
                                |
                    +-----------+------------+
                    | Live LLM tool loop     | Scripted deterministic flow
                    | chooses/refines search | fixed transparent sampling
                    +-----------+------------+
                                | calls bounded local tools
          +---------------------+--------------------------+
          | load profile | inspect system | propose plans  |
          | evaluate candidates | compare results         |
          +---------------------+--------------------------+
                                v
        Analytical simulator: latency, throughput, memory,
        SLA feasibility, stage bottlenecks and trade-offs
                                |
                    verified report + JSON + trace
```

The user supplies a Hugging Face `config.json` (architecture metadata only; model weights are not needed), GPU system and count, input/output sequence lengths, optional cached prefix, precision assumptions, TTFT/TPOT limits and candidate search ranges. The system returns the best evaluated feasible configuration when one exists, including mode, TP/PP/DP representation, batch/workers, estimated output tokens per second per budget GPU, user-level decode rate, TTFT, TPOT, memory use, concurrency capacity and bottleneck. It also writes machine-readable results and an agent trace. The five local tools are `load_model_profile`, `inspect_system`, `propose_deployments`, `evaluate_deployment`, and `compare_results`. Tool schemas and candidate-ID checks constrain the LLM: it cannot submit invented plans or overwrite calculated metrics.

The analytical model uses model dimensions and GPU profiles to estimate compute, memory traffic, KV-cache footprint, communication and stage capacity. It checks memory/SLA feasibility, models aggregate and disaggregated plans, and ranks candidates by estimated output throughput per GPU budget. Effective utilization, scheduler overhead, collective latency and KV-transfer bandwidth are visible inputs or documented defaults. The program does not launch an engine or guarantee a production deployment.

## Target metrics and evaluation method

The product targets are (1) produce a deployment candidate satisfying user-specified TTFT and TPOT limits and memory constraints, (2) maximize estimated output tokens/s per budget GPU among evaluated feasible candidates, and (3) explain bottlenecks and trade-offs. Because each user supplies their own SLA, there is no single global TTFT/TPOT target. For estimator validation, the desired metrics are measured-versus-predicted TTFT and TPOT error and the ordering of deployment configurations under comparable conditions. A planned five-model throughput-ranking metric could not be computed: the available source has no measured output tokens/s/GPU for a matched five-model cohort.

Instead, the project filters NVIDIA AIConfigurator's `silicon_sample.csv` into 778 non-WIP Dense aggregate records, spanning H100/H200 SXM, BF16/FP8, and three base model sizes: Llama 3.1 8B, Llama 3.1 70B and Qwen3 32B. Llama 3.1 8B BF16 and FP8 are distinct checkpoint/precision identities but not separate base model sizes. The evaluation runs the simulator at each row's measured TP/PP/DP and batch size, rather than comparing a measured plan against a separately optimized plan. It computes absolute and percentage errors for TTFT/TPOT. Since the source does not provide request latency, the evaluation derives it as `TTFT + (OSL - 1) × TPOT`, then compares pairwise latency order only within groups matching model, GPU, backend/version, precision and workload.

All 778 eligible records were evaluated; none were skipped. TTFT MAE was 5,956.04 ms, with MAPE 685.9% and median absolute percentage error 449.2%. TPOT MAE was 12.62 ms, MAPE 47.8%, and median absolute percentage error 49.2%. Thus the project did not meet an implicit goal of accurate absolute latency prediction. A relative result was stronger: 5,714 of 6,016 non-tied configuration pairs agreed in latency order (94.98%), and the lowest-latency candidate was identified in 50 of 61 groups (81.97%). These rankings suggest possible value for coarse shortlisting, but they do not establish generalization or real-world utility on new workloads.

## Critical reflection and limitations

The largest technical lesson is that a physically motivated model is not automatically calibrated. Backend kernels, continuous batching, scheduling, graph capture and cache behavior materially affect measurements. Simplified roofline and stage-capacity assumptions miss enough behavior to produce very large TTFT errors. The high within-group rank agreement must be reported alongside—not instead of—absolute errors. Sweep observations are correlated: 6,016 pair comparisons are not 6,016 independent experiments. Ranking comparisons are restricted to matched groups. BF16 KV and zero-prefix assumptions fill source gaps but add mismatch. No confidence intervals or independent holdout split are claimed.

Scope is another rough edge. The planner is designed for Dense full-attention models and rejects MoE, MLA, hybrid attention and other unsupported architectures. It estimates concurrency from tested decode slots and does not model queueing or p99 latency. Scripted mode samples a bounded search and can miss a better plan; live mode is also bounded by turn, token and evaluation budgets and may incur API cost. Neither mode substitutes for measurements with the intended serving engine, software version and workload.

An important pre-submission check is to make the demo configuration consistent with this supported scope. The checked-in `config.toml` currently selects `Qwen3.8-27B`, whose configuration declares a Qwen3.5 hybrid architecture, and sets live mode. That target is outside the stated Dense/full-attention support and may fail model validation; live mode also requires a working API key. For a reliable recorded demo, select a supported Dense config, use scripted mode or verify the live credentials beforehand, run the project tests, and capture the actual terminal and result files. Do not present an old output artifact as proof that the current config ran successfully.

## Future work and reproducibility

The most valuable next step is prospective calibration on controlled deployments using one serving stack. Record GPU, model revision, precision, KV dtype, cache behavior, ISL/OSL, concurrency and software versions; reserve workloads as a holdout. Tune interpretable parameters, report uncertainty and compare simple baselines. A matched Dense cohort with measured throughput would enable model ranking. Further work could add queueing effects and searches optimizing cost or energy.

Reproduction instructions are in `README.md`. Run `python -m unittest discover -s tests -v` for the planner tests and `python evaluation/run_silicon_validation.py` to regenerate row-level predictions and aggregate metrics without an API key or model weights. The filtered source data, importer, metric definitions, assumptions and provenance are documented in `evaluation/README.md` and `THIRD_PARTY_NOTICES.md`; outputs are stored under `outputs/evaluation/`. The report's central conclusion is intentionally bounded: LLM-DeployAgent demonstrates a legible agent-plus-tools workflow and promising within-group ordering on this dataset, but its current absolute latency estimates are not reliable enough to make unverified deployment promises.
