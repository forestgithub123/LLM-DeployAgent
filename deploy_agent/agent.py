"""Transparent scripted workflow and bounded live LLM tool loop."""
from dataclasses import asdict, dataclass
import json
import os
from typing import Any, Callable

from .domain import Request, Run, positive_int
from .tools import DeploymentTools, pareto_frontier, rank, tool_schemas


@dataclass(frozen=True)
class AgentSettings:
    mode: str = "scripted"
    model: str = "gpt-4.1-mini"
    provider: str = "auto"
    base_url: str | None = None
    max_turns: int = 16
    max_evaluations: int = 2048
    max_output_tokens: int = 6000
    max_total_tokens: int = 150000
    timeout_seconds: float = 60.0
    input_usd_per_million: float | None = None
    output_usd_per_million: float | None = None
    scripted_candidates_per_mode: int = 12

    def __post_init__(self) -> None:
        if self.mode not in ("scripted", "live", "baseline", "llm"):
            raise ValueError("run.mode must be scripted or live")
        if self.provider not in ("auto", "openai", "openrouter", "chat"):
            raise ValueError("agent.provider must be auto, openai, openrouter or chat")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("agent.model must be a non-empty string")
        for key in ("max_turns", "max_evaluations", "max_output_tokens", "max_total_tokens", "scripted_candidates_per_mode"):
            positive_int(getattr(self, key), f"agent.{key}")
        if self.scripted_candidates_per_mode > 128:
            raise ValueError("agent.scripted_candidates_per_mode cannot exceed 128")
        import math
        for key in ("timeout_seconds", "input_usd_per_million", "output_usd_per_million"):
            value = getattr(self, key)
            if value is None and key != "timeout_seconds":
                continue
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"agent.{key} must be a finite non-negative number")
        if self.timeout_seconds == 0:
            raise ValueError("timeout_seconds must be positive")


PROMPT = """You are the LLM-DeployAgent deployment planner. Use five tools to investigate a
fixed model, GPU budget and latency targets. Do not invent or adjust tool metrics.
Load model and hardware first. Propose and evaluate both agg and disagg where valid.
Start with small coarse samples (limit 8-16), then use observed SLA violations,
memory and stage bottlenecks to choose narrower TP/PP/batch/worker-count regions.
For example, reduce batches for TTFT/TPOT, increase TP for compute/weight bottlenecks,
increase PP only for memory (no assumed latency speedup), and rebalance P/D workers
when their request capacities differ. These are hypotheses: verify with tools.
Compare results after evaluation and before finishing. Refine at least once based
on observed results where the evaluation/turn budget allows. Never change SLA or
precision, fetch files, or claim a global optimum. Configuration arrays are hard
search bounds; [] means all. IDs must come from propose_deployments.
The objective is maximum output tokens/s per budget GPU among feasible tested plans.
EP=1 for dense models; independent replicas are deployment-level DP, not another
factor inside a worker. Concurrency is tested decode slots, not a proven maximum.
If no tested plan is feasible, say so and explain observed constraints; this does
not prove no possible deployment exists. Final response should be concise Chinese:
your search rationale, observed tradeoffs, limitations and next measurement needed.
The host renders the final numeric recommendation from verified tool results, not
your prose. Do not give unverified performance numbers or pretend to deploy a server.
"""


class DeploymentAgent:
    def __init__(self, settings: AgentSettings | None = None, client=None, progress: Callable[[str], None] | None = None):
        self.settings = settings or AgentSettings()
        self.client = client
        self.progress = progress or (lambda _: None)

    def _run_scripted(self, tools: DeploymentTools) -> None:
        """Transparent, deterministic tool workflow; deliberately makes no API calls."""
        self.progress("[1/5] load_model_profile：读取模型结构与参数量")
        model = tools.call("load_model_profile", {})
        if "error" in model:
            raise ValueError(model["error"])
        self.progress(f"      模型已载入：{model['name']}，约 {model['parameters_estimate']:,} 参数；MoE={model['is_moe']}")

        self.progress("[2/5] inspect_system：读取 GPU 规格与资源预算")
        hardware = tools.call("inspect_system", {})
        if "error" in hardware:
            raise ValueError(hardware["error"])
        self.progress(f"      硬件已载入：{hardware['name']} × {hardware['total_gpus']}，MFU 假设 {hardware['effective_mfu']:.2f}")

        ids: list[str] = []
        limit = self.settings.scripted_candidates_per_mode
        for label, mode in (("3a/5", "agg"), ("3b/5", "disagg")):
            self.progress(f"[{label}] propose_deployments：为 {mode} 生成候选（每种最多抽样 {limit} 个）")
            proposed = tools.call("propose_deployments", {
                "mode": mode, "tp_sizes": [], "pp_sizes": [], "batch_sizes": [],
                "prefill_tp_sizes": [], "prefill_batch_sizes": [], "worker_counts": [], "limit": limit,
            })
            if "error" in proposed:
                raise ValueError(proposed["error"])
            mode_ids = [candidate["id"] for candidate in proposed["candidates"]]
            ids.extend(mode_ids)
            self.progress(f"      合法组合 {proposed['matching_count']} 个；本轮确定性抽样 {len(mode_ids)} 个")

        if not ids:
            raise ValueError("搜索范围内没有可评估的部署候选")
        if len(ids) > tools.max_evaluations:
            raise ValueError(f"scripted 候选共 {len(ids)} 个，超过 max_evaluations={tools.max_evaluations}；降低 scripted_candidates_per_mode")
        self.progress(f"[4/5] evaluate_deployment：用本地确定性模型评估 {len(ids)} 个候选，不调用 API")
        evaluated = tools.call("evaluate_deployment", {"candidate_ids": ids})
        if "error" in evaluated:
            raise ValueError(evaluated["error"])
        self.progress(f"      已评估 {evaluated['evaluated_count']} 个；SLA 可行 {sum(r['feasible'] for r in evaluated['results'])} 个")

        self.progress("[5/5] compare_results：按 SLA、吞吐/GPU 和 Pareto 关系比较")
        comparison = tools.call("compare_results", {})
        if "error" in comparison:
            raise ValueError(comparison["error"])
        best = comparison["best"]
        if best:
            self.progress(f"      最佳已测方案：{best['id']}，{best['tokens_s_per_gpu']:.2f} tokens/s/GPU，TTFT {best['ttft_ms']:.2f} ms，TPOT {best['tpot_ms']:.2f} ms")
        else:
            self.progress("      当前抽样中没有满足 SLA 的方案；这不代表整个搜索空间无解")
        self.progress("      scripted 是透明的规则/采样流程，不代表 LLM 自主推理；API 调用数为 0。")

    def _client(self):
        if self.client is not None:
            return self.client
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("live mode requires OPENAI_API_KEY or OPENROUTER_API_KEY; it will not silently fall back to scripted")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional SDK: python3 -m pip install -r requirements-llm.txt") from exc
        provider = self.settings.provider
        if provider == "auto":
            provider = "openrouter" if api_key.startswith("sk-or-") else "openai"
        base_url = self.settings.base_url or os.environ.get("OPENAI_BASE_URL")
        if provider == "openrouter" and not base_url:
            base_url = "https://openrouter.ai/api/v1"
        client_kwargs = {"api_key": api_key, "timeout": self.settings.timeout_seconds, "max_retries": 1}
        if base_url:
            client_kwargs["base_url"] = base_url
        if provider == "openrouter":
            client_kwargs["default_headers"] = {
                "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "http://localhost"),
                "X-Title": os.environ.get("OPENROUTER_APP_NAME", "LLM-DeployAgent"),
            }
        self._provider = provider
        self.client = OpenAI(**client_kwargs)
        return self.client

    @staticmethod
    def _chat_schemas():
        # Responses API puts function fields at the top level; Chat Completions
        # nests them under `function`.
        return [{"type": "function", "function": {
            "name": item["name"], "description": item.get("description", ""),
            "parameters": item["parameters"], "strict": item.get("strict", True),
        }} for item in tool_schemas()]

    def _run_chat(self, tools, request, usage):
        messages: list[Any] = [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": json.dumps({"request": asdict(request), "budgets": asdict(self.settings)})},
        ]
        analysis, stop_reason = "", "turn_budget_exhausted"
        for turn in range(self.settings.max_turns):
            response = self.client.chat.completions.create(
                model=self.settings.model, messages=messages, tools=self._chat_schemas(),
                tool_choice="auto", parallel_tool_calls=False,
                max_tokens=self.settings.max_output_tokens,
            )
            usage["api_calls"] += 1
            response_usage = getattr(response, "usage", None)
            usage["input_tokens"] += getattr(response_usage, "prompt_tokens", 0) or 0
            usage["output_tokens"] += getattr(response_usage, "completion_tokens", 0) or 0
            message = response.choices[0].message
            calls = list(getattr(message, "tool_calls", None) or [])
            assistant = {"role": "assistant", "content": message.content or ""}
            if calls:
                assistant["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]
            messages.append(assistant)
            tools.trace.append({"turn": turn + 1, "source": "llm", "api_style": "chat_completions",
                                "text": message.content or "", "tool_calls": [{"name": c.function.name, "arguments": c.function.arguments} for c in calls],
                                "usage": {"input_tokens": getattr(response_usage, "prompt_tokens", 0) or 0, "output_tokens": getattr(response_usage, "completion_tokens", 0) or 0}})
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object")
                    output = tools.call(call.function.name, arguments)
                except (ValueError, TypeError) as exc:
                    output = {"error": str(exc)}
                    tools.trace.append({"tool": call.function.name, "observation": output})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(output, ensure_ascii=False)})
            if usage["input_tokens"] + usage["output_tokens"] >= self.settings.max_total_tokens:
                stop_reason = "token_budget_exhausted"
                break
            if not calls:
                required_modes = {mode for mode in ("agg", "disagg") if next(tools.candidate_plans(mode, [], [], [], [], [], []), None)} if tools.model and tools.gpu else {"agg", "disagg"}
                observed_modes = {result.plan.mode for result in tools.results.values()}
                compared = any(step.get("tool") == "compare_results" for step in tools.trace)
                if tools.results and required_modes <= observed_modes and compared:
                    analysis, stop_reason = message.content or "", "completed"
                    break
                messages.append({"role": "user", "content": "Incomplete evidence: load profiles, evaluate candidates in every valid mode, and call compare_results before finishing. Do not invent a result."})
        return analysis, stop_reason

    def run(self, request: Request) -> Run:
        settings = self.settings
        # Baseline enumerates up to a separate hard safety cap; it is not an LLM agent.
        is_live = settings.mode in ("live", "llm")
        tools = DeploymentTools(request, settings.max_evaluations if is_live else 50000)
        usage: dict[str, Any] = {"api_calls": 0, "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": None}
        analysis = ""
        stop_reason = "completed"
        if not is_live and settings.mode in ("scripted",):
            self._run_scripted(tools)
        elif not is_live:
            for tool in ("load_model_profile", "inspect_system"):
                observation = tools.call(tool, {})
                if "error" in observation:
                    raise ValueError(observation["error"])
            count = 0
            for mode in ("agg", "disagg"):
                for plan in tools.candidate_plans(mode, [], [], [], [], [], []):
                    if count == tools.max_evaluations:
                        raise ValueError("Baseline search exceeds 50,000 plans; narrow config search arrays")
                    tools.plans[plan.identifier] = plan
                    tools.results[plan.identifier] = tools._simulator().evaluate(plan)
                    count += 1
            tools.trace.append({"tool": "baseline_exhaustive_search", "arguments": {}, "observation": {"evaluated_count": count}, "source": "host_baseline_not_llm"})
        else:
            client = self._client()
            if getattr(self, "_provider", "openai") in ("openrouter", "chat"):
                analysis, stop_reason = self._run_chat(tools, request, usage)
            else:
                history: list[Any] = [{"role": "user", "content": json.dumps({"request": asdict(request), "budgets": asdict(settings)})}]
                stop_reason = "turn_budget_exhausted"
                for turn in range(settings.max_turns):
                    response = client.responses.create(
                        model=settings.model, instructions=PROMPT, input=history,
                        tools=tool_schemas(), parallel_tool_calls=False,
                        max_output_tokens=settings.max_output_tokens, store=False,
                    )
                    usage["api_calls"] += 1
                    response_usage = getattr(response, "usage", None)
                    for key in ("input_tokens", "output_tokens"):
                        usage[key] += getattr(response_usage, key, 0) or 0
                    outputs = list(response.output)
                    history.extend(item.model_dump(exclude_none=True) for item in outputs)
                    calls = [item for item in outputs if item.type == "function_call"]
                    tools.trace.append({"turn": turn + 1, "source": "llm", "api_style": "responses",
                                        "response_status": getattr(response, "status", None),
                                        "text": getattr(response, "output_text", ""),
                                        "tool_calls": [{"name": item.name, "arguments": item.arguments} for item in calls],
                                        "usage": {key: getattr(response_usage, key, 0) or 0 for key in ("input_tokens", "output_tokens")}})
                    for call in calls:
                        try:
                            arguments = json.loads(call.arguments)
                            if not isinstance(arguments, dict):
                                raise ValueError("Tool arguments must be a JSON object")
                            output = tools.call(call.name, arguments)
                        except (ValueError, TypeError) as exc:
                            output = {"error": str(exc)}
                            tools.trace.append({"tool": call.name, "observation": output})
                        history.append({"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(output)})
                    if usage["input_tokens"] + usage["output_tokens"] >= settings.max_total_tokens:
                        stop_reason = "token_budget_exhausted"
                        break
                    if not calls:
                        required_modes = {mode for mode in ("agg", "disagg") if next(tools.candidate_plans(mode, [], [], [], [], [], []), None)} if tools.model and tools.gpu else {"agg", "disagg"}
                        observed_modes = {result.plan.mode for result in tools.results.values()}
                        compared = any(step.get("tool") == "compare_results" for step in tools.trace)
                        if tools.results and required_modes <= observed_modes and compared:
                            analysis = getattr(response, "output_text", "") or ""
                            stop_reason = "completed"
                            break
                        history.append({"role": "user", "content": "Incomplete evidence: load profiles, evaluate candidates in every valid mode, and call compare_results before finishing. Do not invent a result."})
            if tools.model is None or tools.gpu is None:
                raise RuntimeError("LLM did not load required model/hardware within its budget; inspect model tool-calling support")
        comparison = tools.call("compare_results", {})
        tools.trace[-1]["source"] = "host_final_verification"
        all_results = list(tools.results.values())
        feasible = rank(all_results)
        if settings.input_usd_per_million is not None and settings.output_usd_per_million is not None:
            usage["estimated_cost_usd"] = (usage["input_tokens"] * settings.input_usd_per_million + usage["output_tokens"] * settings.output_usd_per_million) / 1e6
        run = Run(
            request=request, model=tools.model, gpu=tools.gpu, results=all_results,
            best=feasible[0] if feasible else None, pareto=pareto_frontier(all_results),
            best_by_mode={mode: next((result for result in feasible if result.plan.mode == mode), None) for mode in ("agg", "disagg")},
            trace=tools.trace, agent_mode=settings.mode, stop_reason=stop_reason, usage=usage, analysis=analysis,
        )
        from .report import format_report
        run.report = format_report(run)
        return run
