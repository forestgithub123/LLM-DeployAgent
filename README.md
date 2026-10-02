# LLM-DeployAgent

**An explainable planning assistant for choosing a first-pass LLM deployment configuration.**

**一个可解释的部署规划助手：** 输入模型结构、GPU 预算、工作负载和延迟要求，比较一批候选方案，并解释哪些方案符合约束。它是“部署前的规划工具”，不是实际运行模型的推理服务。

> **Before you run / 运行前请先看**
>
> The current `config.toml` selects `live` mode and points to the bundled Qwen3.8-27B architecture file. That architecture uses features this estimator does not support, so it will be rejected before an LLM call. To try the project, use a supported dense full-attention model config (the bundled Qwen3 32B example is at `evaluation/model_configs/qwen3-32b.json`) and start with `scripted` mode. Live mode may incur API charges.
>
> 当前 `config.toml` 选择了 `live` 模式，并指向项目中附带的 Qwen3.8-27B 结构文件。该模型包含本估算器暂不支持的结构，因此程序会在调用大模型前拒绝运行。初次体验请改用受支持的 Dense 全注意力模型配置（项目附带的 Qwen3 32B 示例：`evaluation/model_configs/qwen3-32b.json`），并先使用 `scripted` 模式。`live` 模式会调用 API，可能产生费用。

## 1. What this project does / 项目做什么

When a person sends a prompt to a language model, the computer first processes the prompt and then generates the answer one token at a time. A **token** is a small piece of text; it may be a word, part of a word, or punctuation. Large models need substantial GPU memory and compute, so engineers must decide how many GPUs to use, how to split the work, and how much work to process together.

用户向大语言模型提问时，计算机先处理输入，再逐个生成回答。模型规模越大，对 GPU 显存和计算能力的要求越高，因此部署前要估算：需要多少张 GPU、怎样拆分计算、一次处理多少请求，以及能否满足响应时间要求。

This project helps narrow those choices. It reads a local model architecture file and the assumptions in `config.toml`, proposes aggregated and disaggregated deployment candidates, estimates their memory and latency, filters candidates that fail the configured limits, and ranks the remaining tested candidates by estimated output tokens per second per GPU.

本项目会读取本地模型结构文件和 `config.toml` 中的条件，提出“输入处理与回答生成共用一组 GPU”或“分开处理”的候选方案，估算显存和延迟，过滤掉不满足条件的方案，并对已评估且符合限制的候选按每张预算 GPU 的估算输出速度排序。

**Important:** it does not download model weights, start a serving engine, send test requests to a GPU, or certify production performance. It produces analytical estimates. Treat its recommendations as candidates for later real benchmarking, not as a guarantee.

**重要：** 本项目不会下载模型权重、启动推理服务、在 GPU 上真实压测，也不会保证生产环境表现。它给出的是数学估算和候选方案；真正部署前仍应使用目标推理引擎和实际硬件进行压测。

## 2. A few terms in plain language / 术语解释

| Term | Plain English | 中文说明 |
| --- | --- | --- |
| Model config | A small JSON file describing the model's layer and attention sizes. It is not the model itself. | 描述模型层数、隐藏维度、注意力头数等结构的 JSON 文件；它不是模型权重。 |
| GPU budget | The maximum number of GPUs available for this deployment. | 本次部署最多可以使用的 GPU 数量。 |
| ISL / OSL | Input Sequence Length / Output Sequence Length, measured in tokens. | 输入长度 / 输出长度，单位是 token。 |
| Prefix | The part of the prompt assumed to be already cached and reusable. | 假设已经缓存、可复用的输入前缀长度。设为 0 表示不使用前缀缓存。 |
| TTFT | Time To First Token: time until the first answer token appears. | 首 token 延迟：从提交请求到看到回答第一个 token 的时间。 |
| TPOT | Time Per Output Token: average time between generated tokens. | 每输出 token 时间：回答生成过程中，平均每个 token 之间的耗时。 |
| SLA target | A response-time limit the user wants the deployment to meet. | 用户设定的服务目标，例如 TTFT 不超过 300 ms。它是目标，不是本项目保证达到的结果。 |
| TP / PP | Tensor Parallelism splits layer calculations; Pipeline Parallelism assigns layer groups to different GPUs. | 张量并行把一层的计算拆到多张 GPU；流水线并行把不同层组放到不同 GPU。 |
| DP / replicas | Independent copies of a worker group that can serve separate requests. | 数据并行/副本：部署多个相同工作组，让它们分别处理请求。 |
| Aggregated (agg) | The same worker group handles prompt processing and answer generation. | 聚合部署：同一组 worker 既处理输入，也生成回答。 |
| Disaggregated (disagg) | Separate worker groups handle prompt processing and answer generation. | 解耦部署：输入处理（prefill）和回答生成（decode）由不同 worker 组承担。 |
| Concurrency capacity | Decode slots counted by the simulator for a tested candidate. | 模拟器为已测候选计算的 decode 槽位数；它不是经过排队、突发流量或 P99 延迟验证的生产并发上限。 |

## 3. How the agent works / Agent 如何工作

The five local tools have fixed responsibilities. In `scripted` mode the program calls them in a transparent, repeatable order. In `live` mode a language model can choose which candidates to inspect next and refine the search after seeing results. In both modes, the local analytical estimator calculates the numbers; the LLM is not allowed to invent performance values.

五个本地工具各司其职。`scripted` 模式按固定、可重复的流程调用；`live` 模式由大模型观察中间结果后决定下一步测哪些候选、是否缩小搜索范围。两种模式下，数字都由本地估算器计算，大模型不能编造性能指标。

```text
Your config.toml + local model config
你的配置文件 + 本地模型结构文件
                  │
                  ▼
       Validate inputs / 检查输入
                  │
                  ▼
 Scripted planner or live LLM / 固定流程或 LLM 规划
                  │
                  ├── 1. Load model structure / 读取模型结构
                  ├── 2. Inspect GPU profile / 读取 GPU 参数
                  ├── 3. Propose deployment candidates / 提出候选
                  ├── 4. Estimate candidates / 估算候选
                  └── 5. Compare results / 比较结果
                  │
                  ▼
 report.txt + result.json + agent_trace.json
```

| Tool / 工具 | What it does / 作用 |
| --- | --- |
| `load_model_profile` | Reads architecture values from the local JSON and estimates parameter count. / 读取本地 JSON 中的结构信息并估算参数量。 |
| `inspect_system` | Loads the selected GPU profile and records the assumed memory, bandwidth and compute efficiency. / 读取所选 GPU 的显存、带宽和计算效率假设。 |
| `propose_deployments` | Creates valid TP/PP/batch/worker combinations for `agg` or `disagg`. / 为两种部署模式生成满足资源约束的 TP、PP、batch 和 worker 组合。 |
| `evaluate_deployment` | Estimates latency, throughput, memory use, bottleneck and SLA feasibility for selected candidates. / 估算所选候选的延迟、吞吐、显存、瓶颈和目标是否满足。 |
| `compare_results` | Ranks feasible evaluated candidates and reports alternatives and trade-offs. / 排序符合限制的已评估方案，并汇总备选项与取舍。 |

For dense models, EP (Expert Parallelism) is fixed to 1 because there are no routed experts. DP is represented by the number of independent replicas. The search arrays in the config are hard limits: the planner cannot choose a TP size or batch size that is not listed there.

对于 Dense（稠密）模型，EP 固定为 1，因为没有需要分发的专家层。DP 由独立副本数表示。配置中的搜索数组是硬限制：例如 `tp_sizes` 没有 8，Agent 就不会尝试 TP=8。

## 4. Quick start / 快速开始

### Requirements / 运行环境

- Python 3.10 or newer; Python 3.11+ is recommended. No GPU is required for the analytical planner.
- `scripted` mode has no API-key requirement and does not need the optional LLM SDK.
- Python 3.10 needs the `tomli` package for TOML parsing; Python 3.11+ includes `tomllib`.

- Python 3.10 或更高版本，推荐 Python 3.11 及以上。运行解析估算不要求本机有 GPU。
- `scripted` 模式不需要 API key，也不需要可选的大模型 SDK。
- Python 3.10 需要安装 `tomli` 才能读取 TOML 配置；Python 3.11 及以上自带 `tomllib`。

### First run / 首次运行

1. Open `config.toml`.
2. Set `run.mode = "scripted"` for a no-API test.
3. Set `input.model_path` to a supported local Dense full-attention model config. The bundled Qwen3 32B example can be used as a starting point:

1. 打开 `config.toml`。
2. 将 `run.mode` 设为 `"scripted"`，先做不调用 API 的测试。
3. 将 `input.model_path` 改成受支持的 Dense 全注意力模型结构文件。初次运行可先用项目附带的 Qwen3 32B 示例：

```toml
mode = "scripted" # in the [run] section / 位于 [run] 配置段
model_path = "evaluation/model_configs/qwen3-32b.json" # in [input] / 位于 [input] 配置段
```

The paths above are relative to `config.toml`, so they work when you run the project from any terminal directory. Then run:

上面路径相对于 `config.toml` 所在目录，因此从其他终端目录启动也能找到文件。然后运行：

```bash
python run.py
```

No other config argument is needed. Results are written to the directory configured by `[run].output_dir` (the current file uses `outputs/live`). Running again overwrites the three result files in that directory; copy them elsewhere first if you need to preserve an earlier run.

不需要另外指定配置文件。结果写入 `[run].output_dir` 指定的目录（当前配置为 `outputs/live`）。再次运行会覆盖该目录下的三个结果文件；要保留旧结果，请先复制到别处。

### Set up a Python environment / 创建 Python 环境

If your system Python is older than 3.10, create an environment first. Example with Conda:

如果系统 Python 低于 3.10，请先建立新环境。Conda 示例：

```bash
conda create -n deploy-agent python=3.11 -y
conda activate deploy-agent
python run.py
```

If using Python 3.10, install its small TOML compatibility package:

若使用 Python 3.10，另安装 TOML 兼容包：

```bash
python -m pip install tomli
```

## 5. Choose a model / 更换测试模型

The program needs the architecture `config.json`, not the large weight files. Download the config file for the **exact model variant** you want to study, place it under `src/data/`, then set its relative path in `config.toml`:

程序只需要模型结构用的 `config.json`，不需要体积很大的权重文件。请下载你要研究的**确切模型版本**对应的配置文件，放入 `src/data/`，再在 `config.toml` 中填写相对路径：

```text
LLM-DeployAgent/
├── config.toml
└── src/data/MyModel/config.json
```

```toml
[input]
model_path = "src/data/MyModel/config.json"
```

The supported architecture is currently dense, full-attention models using standard MHA or GQA fields. Mixture-of-Experts (MoE), MLA, hybrid/linear attention, sliding-window attention, and multimodal models are rejected. A model name such as `Qwen/SomeModel` is not enough: the program does not look up model names online. It needs a local JSON architecture file.

目前支持的是采用标准 MHA 或 GQA 结构字段的 Dense 全注意力模型。MoE（混合专家）、MLA、混合/线性注意力、滑动窗口注意力和多模态模型会被拒绝。只填写 `Qwen/某个模型名` 不够：程序不会根据模型名上网查询，必须提供本地 JSON 结构文件。

The bundled `src/data/Qwen/Qwen3.8-27B.json` is a real model architecture file, but that particular model is hybrid and is **not supported by the current estimator**. Use an architecture that meets the support conditions above, or extend and validate the estimator before using an unsupported model.

项目附带的 `src/data/Qwen/Qwen3.8-27B.json` 是真实模型结构文件，但该模型属于混合结构，**当前估算器不支持**。请换成符合上述条件的模型，或先扩展并验证估算器后再尝试。

## 6. Edit the input / 修改输入

Edit `[input]` in `config.toml`. The comments in that file describe every option. Common changes are:

在 `config.toml` 的 `[input]` 配置段中修改输入。每个字段旁都有注释，常改的内容如下：

| Setting | Meaning / 含义 | Example / 示例 |
| --- | --- | --- |
| `model_path` | Local architecture JSON file / 本地模型结构 JSON | `"src/data/MyModel/config.json"` |
| `system` | GPU profile used by the estimate / 估算使用的 GPU 型号 | `"h200_sxm"` |
| `total_gpus` | Maximum GPU count available / 可用 GPU 总数 | `8` |
| `isl`, `osl` | Input and generated-output lengths, in tokens / 输入和输出 token 长度 | `4000`, `500` |
| `prefix` | Cached input prefix length; must be less than `isl` / 假设已缓存的输入前缀，必须小于 `isl` | `500` |
| `ttft`, `tpot` | Maximum desired latency, in milliseconds / 延迟目标，单位毫秒 | `300.0`, `10.0` |
| `precision`, `kv_precision` | Assumed weight and KV-cache formats: `fp8` or `bf16` / 假设的权重和 KV 缓存精度 | `"fp8"` |
| `tp_sizes`, `pp_sizes` | Parallelism values the planner may try / 允许尝试的并行度 | `[1, 2, 4, 8]` |
| `batch_sizes` | Decode batch sizes it may try / 允许尝试的 decode 批大小 | `[1, 2, 4, 8, 16]` |

An example request in plain language: “Use this local model structure, assume 8 H200 GPUs, prompts of about 4,000 tokens and answers of 500 tokens. Find tested candidates whose estimated first-token latency is at most 300 ms and average output-token time is at most 10 ms.”

通俗示例：“使用这个本地模型结构，假设有 8 张 H200，输入约 4,000 个 token、回答约 500 个 token。请找出估算首 token 延迟不超过 300 毫秒、平均每个输出 token 不超过 10 毫秒的候选方案。”

GPU names currently available: `H100`, `H20`, `H800`, `H200`, `PRO5000`; common `_SXM` aliases are also accepted. The GPU count means a budget for the plan, not a check of hardware physically installed on your computer.

当前内置 GPU 型号：`H100`、`H20`、`H800`、`H200`、`PRO5000`，也接受常用的 `_SXM` 别名。GPU 数量表示你给规划器的资源预算，并不表示程序检查了本机实际安装的 GPU。

## 7. `scripted` and `live` modes / 两种运行模式

### `scripted` — no LLM call / 不调用大模型

The program prints each fixed step and samples a limited number of candidates per deployment mode. It is useful for learning the workflow, checking inputs, automated tests and runs without network/API access. It is not an LLM making decisions.

程序会打印固定的工具调用步骤，并对每种部署模式抽样有限数量的候选。适合初次学习、检查输入、自动化测试，以及没有网络或 API 的环境。它不是由大模型进行决策。

```toml
[run]
mode = "scripted"
```

### `live` — LLM-assisted search / 大模型辅助搜索

In this mode the LLM observes tool results and can choose a next search region—for example, testing smaller batches after seeing a latency issue. The LLM does not calculate or certify the performance numbers; local tools do that. Live calls can use tokens and may be billed by your provider.

此模式下，大模型会查看工具结果，并可以选择下一步搜索范围；例如，发现延迟偏高后尝试更小的 batch。性能数字仍由本地工具计算，大模型不负责计算或认证。API 会消耗 token，并可能按服务商规则收费。

Install the optional dependency and set the key in your **terminal**, not in `config.toml`:

安装可选依赖，并在**终端**设置密钥，不要把密钥写进 `config.toml`：

```bash
python -m pip install -r requirements-llm.txt
export OPENAI_API_KEY="your-key"       # OpenAI
# or / 或者:
export OPENROUTER_API_KEY="your-key"   # OpenRouter
```

Then set `[run].mode = "live"` and choose `[agent].model`. `agent.model` is the model that plans the search; `input.model_path` is the model being analyzed. They are different. Do not commit API keys or paste them into screenshots, logs or reports. A key error (HTTP 401) usually means the key/provider/base URL combination is incorrect.

然后将 `[run].mode` 设为 `"live"`，并选择 `[agent].model`。`agent.model` 是负责规划搜索的大模型；`input.model_path` 是正在分析的目标模型，两者不同。不要提交 API key，也不要把它放入截图、日志或报告。若遇到 HTTP 401，通常是 key、服务商或 base URL 不匹配。

## 8. Read the outputs / 查看运行结果

The configured output folder receives three files:

每次运行会在配置的结果文件夹中生成三个文件：

| File / 文件 | What it contains / 内容 |
| --- | --- |
| `report.txt` | Human-readable summary: best tested candidate, whether it met the targets, trade-offs and cautions. / 可读摘要：最佳已测候选、是否满足目标、取舍和注意事项。 |
| `result.json` | Full machine-readable input, candidates, estimates and final comparison. / 完整机器可读数据：输入、候选、估算和比较结果。 |
| `agent_trace.json` | Ordered tool calls and observations; in live mode, also the LLM's tool choices and token usage. / 工具调用和观察记录；live 模式还包括大模型的工具选择与 token 用量。 |

“Best” means best among the candidates actually evaluated. Scripted mode samples; live mode is limited by its turn, evaluation and token budgets. Neither mode guarantees that the global best deployment was found.

“最佳”仅表示在**实际评估过的候选中**最好。`scripted` 会抽样；`live` 受对话轮数、评估数量和 token 预算限制。两种模式都不保证找到全局最优部署。

## 9. Evaluation and current limitations / 评估结果与当前局限

The project includes 778 NVIDIA AIConfigurator silicon measurements for Dense models. The evaluation compares the estimator with recorded TTFT/TPOT and compares the latency order of plans only inside groups with matching workload and hardware conditions. It does not have measured output-throughput labels for a valid five-model throughput-ranking test.

项目附带 778 条 NVIDIA AIConfigurator Dense 模型实测记录。评估将估算 TTFT/TPOT 与记录值比较，并只在工作负载和硬件条件相同的组内比较延迟排序。现有数据没有实测输出吞吐标签，因此不能据此完成可靠的五模型吞吐排名。

| Validation metric / 评估指标 | Current result / 当前结果 | Plain-language reading / 通俗解释 |
| --- | ---: | --- |
| TTFT MAE | 5,956.04 ms | Absolute first-token timing estimate is poor. / 首 token 绝对延迟误差很大。 |
| TTFT MAPE / median APE | 685.9% / 449.2% | Relative first-token error is also very large. / 首 token 相对误差也很大。 |
| TPOT MAE | 12.62 ms | Average output-token timing still has material error. / 每个输出 token 的平均耗时仍有明显误差。 |
| TPOT MAPE / median APE | 47.8% / 49.2% | Typical relative error is around half the measured value. / 典型相对误差约为实测值的一半。 |
| Pairwise order agreement | 94.98% (5,714 / 6,016 pairs) | Within matched groups, many pairs are ordered correctly; this does not mean their timings are accurate. / 相同条件组内，很多配置的快慢顺序判断正确，但不代表延迟数值准确。 |
| Lowest-latency top-1 agreement | 81.97% (50 / 61 groups) | The measured fastest plan was selected in most matched groups, but not all. / 多数相同条件组中选中了实测最快方案，但仍会选错。 |

The responsible conclusion is that the tool may help with **coarse shortlisting**, but its latency estimates are not reliable enough for deployment guarantees. The measurements are correlated sweep rows, cover only three base model sizes, and reflect specific serving backends. More independent, prospective benchmarks are needed.

合理结论是：本工具可能有助于**粗略筛选候选**，但延迟估算不可靠，不能用来保证部署 SLA。这些记录来自相互关联的参数扫描，只覆盖三种基础模型规模，并受特定推理后端影响；还需要更多独立、前瞻性的实测验证。

To reproduce the bundled evaluation (no API key or model weights required):

复现附带评估（不需要 API key 或模型权重）：

```bash
python evaluation/run_silicon_validation.py
```

Method, filtering rules, assumptions and data provenance are described in [`evaluation/README.md`](evaluation/README.md). Third-party attribution and license notes are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); keep that file with the project when sharing it.

评估方法、数据筛选条件、假设和来源见 [`evaluation/README.md`](evaluation/README.md)。第三方来源与许可说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)；分享或提交项目时请保留该文件。

## 10. Tests / 自动化测试

Run the local unit tests with:

运行本地单元测试：

```bash
python -m unittest discover -s tests -v
```

The tests check the planner, constraints, supported model formats, ranking metrics and the bundled evaluation calculations. They do not benchmark a real inference server.

这些测试检查规划流程、约束条件、支持的模型格式、排序指标和附带评估计算；它们不等同于真实推理服务压测。

## 11. Project files / 项目目录

```text
config.toml                 User inputs and search settings / 用户输入与搜索设置
run.py                      Single launcher / 唯一启动脚本
deploy_agent/               Agent, tools, estimator and report logic / Agent、工具、估算器与报告逻辑
src/data/                   Local model architecture JSON files / 本地模型结构文件
evaluation/                 Reproducible validation scripts and dataset / 可复现评估脚本与数据
outputs/                    Most recent reports, results and traces / 最近一次输出
tests/                      Unit tests / 单元测试
PROJECT_REPORT.md           Course project report / 课程项目报告
THIRD_PARTY_NOTICES.md      Attribution and license notes / 第三方来源与许可说明
```

## 12. Honest scope / 项目能力边界

- The estimator currently supports Dense, standard full-attention MHA/GQA models only. / 当前仅支持 Dense 标准全注意力 MHA/GQA 模型。
- The current GPU profiles and efficiency factors are assumptions, not measurements from your actual machine or serving backend. / GPU 参数和效率系数是模型假设，不是对你本机或推理后端的实时测量。
- Throughput and concurrency are analytical estimates for tested candidates; queueing, traffic bursts, P99 latency, network topology and engine-specific optimizations are not fully modeled. / 吞吐和并发是候选配置的解析估算；排队、流量突发、P99 延迟、网络拓扑和引擎专属优化未被完整建模。
- Passing the estimator's SLA check means only “the estimate is under the configured limit,” not “a real deployment has been proven to meet the SLA.” / 估算结果通过 SLA 检查，只表示“估算值低于配置限制”，不表示真实部署已被证明满足 SLA。

The project's intended use is to make assumptions visible, help users understand deployment trade-offs, and reduce the number of configurations to benchmark next. It should not replace an actual benchmark or a production capacity test.

本项目的目标是把假设说清楚、帮助用户理解部署取舍，并减少下一步需要实测的配置数量；它不能替代真实压测或生产容量测试。
