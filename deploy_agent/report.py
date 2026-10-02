"""AIConfigurator-style output, with project branding and explicit assumptions."""
from .domain import Run
from .tools import rank


def table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(No evaluated configuration satisfies all constraints.)"
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    def render(row):
        return "|" + "|".join(f" {value:<{width}} " for value, width in zip(row, widths)) + "|"
    return "\n".join([line, render(headers), line, *(render(row) for row in rows), line])


def pareto_plot(run: Run) -> str:
    if not run.pareto:
        return "(No feasible Pareto points.)"
    width, height = 60, 10
    max_x = max(r.tokens_s_per_user for r in run.pareto)
    max_y = max(r.tokens_s_per_gpu for r in run.pareto)
    grid = [[" " for _ in range(width)] for _ in range(height)]
    for result in run.pareto:
        x = min(width - 1, int(result.tokens_s_per_user / max_x * (width - 1)))
        y = min(height - 1, int((1 - result.tokens_s_per_gpu / max_y) * (height - 1)))
        grid[y][x] = "*" if result is run.best else "d" if result.plan.mode == "disagg" else "a"
    lines = ["Y: output tokens/s / budget GPU; X: decode tokens/s/user", "a=agg, d=disagg, *=best"]
    for i, row in enumerate(grid):
        lines.append(f"{max_y * (1 - i / (height - 1)):8.1f}|" + "".join(row))
    lines.extend(["        +" + "-" * width, f"         0{' ' * 42}{max_x:.1f}"])
    return "\n".join(lines)


def format_report(run: Run) -> str:
    r, best = run.request, run.best
    feasible = rank(run.results)
    lines = [
        "LLM-DeployAgent Final Results (analytical estimates)", "=" * 64,
        f"Agent: {run.agent_mode}; stop reason: {run.stop_reason}",
        f"Input: {r.model_path} (dense; EP=1), {r.total_gpus} x {run.gpu.name}",
        f"ISL={r.isl}, OSL={r.osl}, cached prefix={r.prefix}; weights={r.precision}, KV={r.kv_precision}",
        f"SLA: TTFT <= {r.ttft:.2f}ms; TPOT <= {r.tpot:.2f}ms",
        f"Evaluated={len(run.results)}; feasible={len(feasible)}; estimator=analytical (not trtllm/vLLM measurements)",
        "", "Overall Best Configuration (among evaluated plans):",
    ]
    if best:
        p = best.plan
        lines.extend([
            f"Best Experiment Chosen: {p.mode}",
            f"Best Throughput: {best.total_tokens_s:,.2f} output tokens/s",
            f"Per-GPU Throughput: {best.tokens_s_per_gpu:,.2f} tokens/s/budget-GPU",
            f"Per-User Throughput (decode): {best.tokens_s_per_user:.2f} tokens/s/user",
            f"TTFT: {best.ttft_ms:.2f}ms; TPOT: {best.tpot_ms:.2f}ms; Request Latency: {best.request_latency_ms:.2f}ms",
            f"GPUs used: {p.used_gpus}/{r.total_gpus}; replicas (DP): {p.replicas}; GPUs/replica: {p.gpus_per_replica}",
            f"Decode: {p.decode.parallel}, batch={p.decode.batch_size}, workers/replica={p.decode_workers}",
            f"Prefill: {p.prefill.parallel}, batch={p.prefill.batch_size}, workers/replica={p.prefill_workers}" if p.prefill else "Prefill/decode share combined workers.",
            f"Recommended decode concurrency capacity: {best.concurrency_capacity}; estimated total active requests: {best.estimated_active_requests:.2f}",
            f"Max tested feasible decode concurrency: {max(item.concurrency_capacity for item in feasible)} (not a proven maximum)",
            f"Bottleneck: {best.bottleneck}; stage memory P/D: {best.prefill_memory_gb:.2f}/{best.decode_memory_gb:.2f}GB/GPU",
        ])
        other = run.best_by_mode["agg" if p.mode == "disagg" else "disagg"]
        if other:
            lines.append(f"{p.mode} is {best.tokens_s_per_gpu / other.tokens_s_per_gpu:.2f}x the best tested {other.plan.mode} throughput.")
    else:
        lines.append("No evaluated deployment satisfies all SLA/memory constraints; no recommendation.")
        closest = sorted(run.results, key=lambda result: (len(result.violations), result.ttft_ms / r.ttft + result.tpot_ms / r.tpot))[:3]
        for result in closest:
            lines.append(f"Closest infeasible: {result.identifier}; " + "; ".join(result.violations))
    lines.extend(["", "Pareto Frontier:", pareto_plot(run), "", "Deployment Details:",
                  "Dense worker GPUs = TP x PP; DP = independent replicas, EP=1.",
                  "Disagg GPUs/replica = P workers x P worker GPUs + D workers x D worker GPUs."])
    for mode in ("agg", "disagg"):
        headers = ["Rank", "tok/s/GPU", "tok/s/user", "TTFT", "TPOT", "latency", "conc", "used", "DP", "GPU/unit"]
        headers += ["parallel", "bs"] if mode == "agg" else ["P workers", "P GPUs", "P parallel", "P bs", "D workers", "D GPUs", "D parallel", "D bs"]
        rows = []
        for index, result in enumerate([v for v in feasible if v.plan.mode == mode][:r.top_k], 1):
            plan = result.plan
            row = [str(index), f"{result.tokens_s_per_gpu:.2f}", f"{result.tokens_s_per_user:.2f}", f"{result.ttft_ms:.2f}", f"{result.tpot_ms:.2f}", f"{result.request_latency_ms:.2f}", str(result.concurrency_capacity), str(plan.used_gpus), str(plan.replicas), str(plan.gpus_per_replica)]
            row += [plan.decode.parallel, str(plan.decode.batch_size)] if mode == "agg" else [str(plan.prefill_workers), str(plan.prefill.gpus), plan.prefill.parallel, str(plan.prefill.batch_size), str(plan.decode_workers), str(plan.decode.gpus), plan.decode.parallel, str(plan.decode.batch_size)]
            rows.append(row)
        lines.extend(["", f"{mode} Top Feasible Configurations (sorted by tokens/s/budget GPU):", table(headers, rows)])
    lines.extend(["", "Evidence & Limits:",
                  "Roofline estimates, conservative sequential PP; not a backend benchmark or deployment command.",
                  "Agg uses a closed-batch cycle. Disagg assumes steady pipelining; throughput is min(P, D, KV-transfer capacity).",
                  "Prefix is a guaranteed cache hit. KV transfer assumes one effective shared link per replica.",
                  "No queueing, continuous-batching simulation, network contention between replicas, or p99 guarantee.",
                  "Per-GPU denominator is the full GPU budget (including idle GPUs); per-user is 1000/TPOT, not total/concurrency.",
                  "TTFT produces token 1; latency = TTFT + (OSL-1)*TPOT. TPOT uses end-of-generation context.",
                  "Validate predictions and ranking with matched serving-stack measurements before production."])
    if run.analysis:
        lines.extend(["", "LLM Analysis (unverified prose; numeric tables above are authoritative):", run.analysis])
    usage = run.usage
    lines.extend(["", "Agent Usage:", f"API calls={usage['api_calls']}; input tokens={usage['input_tokens']}; output tokens={usage['output_tokens']}; estimated cost USD={usage['estimated_cost_usd']}"])
    lines.append("Search/tool observations are recorded in agent_trace.json.")
    return "\n".join(lines)
