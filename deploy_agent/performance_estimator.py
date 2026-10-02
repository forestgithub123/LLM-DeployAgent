"""Analytical estimator for model compute, KV cache, communication and latency.

This project-owned implementation uses SI GB and seconds. It does not run a
serving engine, and its predictions are not serving-stack measurements.
"""
from .domain import GPUProfile, ModelProfile, Plan, Request, Result, Worker


class AnalyticalSimulator:
    def __init__(self, request: Request, model: ModelProfile, gpu: GPUProfile):
        self.r, self.m, self.g = request, model, gpu
        self.weight_bytes = 1 if request.precision == "fp8" else 2
        self.kv_bytes = 1 if request.kv_precision == "fp8" else 2
        self.mfu = request.mfu if request.mfu is not None else gpu.default_mfu

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.m.num_layers * self.m.num_kv_heads * self.m.head_dim * self.kv_bytes

    def validate(self, plan: Plan) -> None:
        r, m = self.r, self.m
        if plan.mode not in ("agg", "disagg") or plan.replicas <= 0:
            raise ValueError("invalid mode or replica count")
        if plan.mode == "agg" and (plan.prefill is not None or plan.prefill_workers != 0 or plan.decode_workers != 1):
            raise ValueError("agg must contain one combined worker per replica")
        if plan.mode == "disagg" and (plan.prefill is None or plan.prefill_workers <= 0 or plan.decode_workers <= 0):
            raise ValueError("disagg requires prefill and decode workers")
        for worker in (plan.prefill, plan.decode):
            if worker is None:
                continue
            if worker.tp not in r.tp_sizes or worker.pp not in r.pp_sizes:
                raise ValueError("parallelism is outside the configured search space")
            if worker.batch_size <= 0:
                raise ValueError("batch must be positive")
            if any(dim % worker.tp for dim in (m.num_attention_heads, m.num_kv_heads, m.intermediate_size)):
                raise ValueError("TP must divide attention heads, KV heads and FFN dimension")
            if m.num_layers % worker.pp:
                raise ValueError("PP must divide layer count")
        if plan.used_gpus > r.total_gpus:
            raise ValueError("GPU budget exceeded")

    def memory_gb(self, worker: Worker, context: int) -> float:
        m, r = self.m, self.r
        # Peak stage: embeddings/head share a stage only for PP=1. They are TP-sharded.
        embedding_copies = (1 if m.tie_word_embeddings else 2) if worker.pp == 1 else 1
        stage_parameters = m.layer_parameters * (m.num_layers // worker.pp) + embedding_copies * m.vocab_size * m.hidden_size
        weights = stage_parameters * self.weight_bytes / worker.tp
        kv = self.kv_bytes_per_token * context * worker.batch_size / (worker.tp * worker.pp)
        return (weights + kv) / 1e9 + r.workspace_gb

    def timing(self, worker: Worker, prefill: bool) -> dict[str, float]:
        m, r, g = self.m, self.r, self.g
        batch = worker.batch_size
        queries = r.isl - r.prefix if prefill else 1
        tokens = batch * queries
        context = r.isl + r.osl - 1
        pairs = batch * (queries * r.prefix + queries * (queries + 1) / 2) if prefill else batch * context
        q = m.num_attention_heads * m.head_dim
        linear_flops = 2 * tokens * (m.layer_parameters * m.num_layers + m.vocab_size * m.hidden_size) / worker.tp
        attention_flops = 4 * pairs * q * m.num_layers / worker.tp
        peak = g.fp8_tflops if r.precision == "fp8" else g.fp16_tflops
        linear_compute = linear_flops / (peak * 1e12 * self.mfu)
        # Attention is conservatively treated as BF16 even for FP8 weights.
        attention_compute = attention_flops / (g.fp16_tflops * 1e12 * self.mfu)
        weight_read = (m.layer_parameters * m.num_layers + m.vocab_size * m.hidden_size) * self.weight_bytes / worker.tp
        kv_read = self.kv_bytes_per_token * batch * (r.isl if prefill else context) / worker.tp
        linear_memory = weight_read / (g.memory_bandwidth_gbps * 1e9)
        attention_memory = kv_read / (g.memory_bandwidth_gbps * 1e9)
        # Two ring all-reduces per layer, BF16 activations. Cross-node TP uses RDMA.
        link = g.nvlink_bandwidth_gbps if worker.tp <= r.gpus_per_node else g.rdma_bandwidth_gbps
        comm = 0.0
        if worker.tp > 1:
            payload = tokens * m.hidden_size * 2
            comm = m.num_layers * 2 * (2 * (worker.tp - 1) / worker.tp * payload / (link * 1e9) + r.collective_latency_us / 1e6)
        pp_link = g.nvlink_bandwidth_gbps if worker.gpus <= r.gpus_per_node else g.rdma_bandwidth_gbps
        comm += (worker.pp - 1) * (tokens * m.hidden_size * 2 / (pp_link * 1e9) + r.collective_latency_us / 1e6)
        # PP stages execute sequentially for this request. No fictitious 1/PP latency gain.
        roofline = max(linear_compute, linear_memory) + max(attention_compute, attention_memory)
        return {
            "compute": (linear_compute + attention_compute) * 1000,
            "memory": (linear_memory + attention_memory) * 1000,
            "communication": comm * 1000,
            "scheduler": r.scheduler_ms,
            "total": (roofline + comm) * 1000 + r.scheduler_ms,
        }

    def evaluate(self, plan: Plan) -> Result:
        self.validate(plan)
        r, g = self.r, self.g
        p = plan.prefill or plan.decode
        prefill_ms = self.timing(p, True)["total"]
        decode = self.timing(plan.decode, False)
        tpot = decode["total"]
        transfer_ms = 0.0
        transfer_capacity = None
        if plan.mode == "disagg":
            # Full prompt KV, including cached prefix, is transferred for each request.
            transfer_bytes = self.kv_bytes_per_token * r.isl
            transfer_ms = transfer_bytes / ((r.kv_transfer_gbps or g.rdma_bandwidth_gbps) * 1e9) * 1000
            transfer_capacity = 1000 / transfer_ms * plan.replicas
            p_capacity = plan.prefill_workers * p.batch_size * 1000 / prefill_ms * plan.replicas
            d_capacity = plan.decode_workers * plan.decode.batch_size * 1000 / (r.osl * tpot) * plan.replicas
            capacities = {"prefill": p_capacity, "decode": d_capacity, "kv_transfer": transfer_capacity}
            bottleneck = min(capacities, key=capacities.get)
            requests_s = min(capacities.values())
            ttft = prefill_ms + transfer_ms + tpot
        else:
            ttft = prefill_ms
            # Closed-batch baseline: combined workers pay for both phases.
            cycle_ms = prefill_ms + (r.osl - 1) * tpot
            requests_s = plan.decode.batch_size * 1000 / cycle_ms * plan.replicas
            p_capacity = plan.decode.batch_size * 1000 / prefill_ms * plan.replicas
            d_capacity = plan.decode.batch_size * 1000 / max(r.osl * tpot, 1e-9) * plan.replicas
            bottleneck = "prefill" if prefill_ms > (r.osl - 1) * tpot else max(("compute", "memory", "communication"), key=lambda key: decode[key])
        latency = ttft + (r.osl - 1) * tpot
        total = requests_s * r.osl
        p_memory = self.memory_gb(p, r.isl)
        d_memory = self.memory_gb(plan.decode, r.isl + r.osl)
        violations = []
        for label, observed, limit in (("TTFT", ttft, r.ttft), ("TPOT", tpot, r.tpot), ("prefill_memory_GB", p_memory, g.memory_gb * r.memory_fraction), ("decode_memory_GB", d_memory, g.memory_gb * r.memory_fraction)):
            if observed > limit:
                violations.append(f"{label}: {observed:.3f} > {limit:.3f}")
        return Result(
            plan=plan, feasible=not violations, ttft_ms=ttft, tpot_ms=tpot,
            request_latency_ms=latency, total_tokens_s=total, tokens_s_per_gpu=total / r.total_gpus,
            tokens_s_per_user=1000 / tpot, requests_s=requests_s,
            concurrency_capacity=plan.decode.batch_size * plan.decode_workers * plan.replicas,
            estimated_active_requests=requests_s * latency / 1000,
            prefill_ms=prefill_ms, kv_transfer_ms=transfer_ms,
            prefill_capacity_requests_s=p_capacity, decode_capacity_requests_s=d_capacity,
            transfer_capacity_requests_s=transfer_capacity, prefill_memory_gb=p_memory,
            decode_memory_gb=d_memory, bottleneck=bottleneck,
            violations=tuple(violations), decode_breakdown_ms=decode,
        )
