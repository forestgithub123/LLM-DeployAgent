"""Five stateful tools; the LLM selects search regions, never invents metrics."""
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
from typing import Any

from .catalog import get_gpu
from .domain import ModelProfile, Plan, Request, Result, Worker, positive_int
from .performance_estimator import AnalyticalSimulator



def load_profile(name: str) -> ModelProfile:
    path = Path(name)
    if path.is_dir():
        path = path / "config.json"
    if not path.is_file():
        expected = Path(__file__).resolve().parents[1] / "src" / "data"
        raise ValueError(
            f"Model config not found: {name!r}. Download the model's Hugging Face config.json "
            f"into {expected} (or a subdirectory) and set input.model_path to that local path. "
            "Model weights are not required."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    data = raw.get("text_config", raw)
    expert_counts = (data.get(key) or 1 for key in ("num_experts", "num_routed_experts", "n_routed_experts"))
    layer_types = data.get("layer_types") or []
    has_nonstandard_attention = any(
        str(layer_type).lower() not in ("full_attention", "attention")
        for layer_type in layer_types
    )
    if ("vision_config" in raw or any(count > 1 for count in expert_counts)
            or "kv_lora_rank" in data or "linear_attn" in str(data).lower()
            or data.get("use_sliding_window") or has_nonstandard_attention):
        raise ValueError("This estimator supports dense full-attention GQA/MHA only; MoE/MLA/hybrid/sliding-window models need another validated estimator")
    fields = {
        "hidden_size": data["hidden_size"], "num_layers": data["num_hidden_layers"],
        "num_attention_heads": data["num_attention_heads"],
        "num_kv_heads": data.get("num_key_value_heads", data["num_attention_heads"]),
        "head_dim": data.get("head_dim", data["hidden_size"] // data["num_attention_heads"]),
        "intermediate_size": data["intermediate_size"], "vocab_size": data["vocab_size"],
        "max_position_embeddings": data.get("max_position_embeddings", 40960),
    }
    for key, value in fields.items():
        positive_int(value, key)
    display_name = raw.get("name_or_path") or raw.get("_name_or_path") or (path.parent.name if path.name == "config.json" else path.stem)
    return ModelProfile(name=display_name, tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
                        source=data.get("source", f"local architecture profile: {path.name}"), **fields)


def rank(results: list[Result]) -> list[Result]:
    return sorted((r for r in results if r.feasible), key=lambda r: (-r.tokens_s_per_gpu, r.ttft_ms, r.identifier))


def pareto_frontier(results: list[Result]) -> list[Result]:
    # Two-objective sweep; equal metric pairs are kept for different deployments.
    ordered = sorted((r for r in results if r.feasible), key=lambda r: (-r.tokens_s_per_gpu, -r.tokens_s_per_user))
    frontier = []
    max_user = -1.0
    previous = None
    for result in ordered:
        pair = (result.tokens_s_per_gpu, result.tokens_s_per_user)
        if result.tokens_s_per_user > max_user or pair == previous:
            frontier.append(result)
            max_user = result.tokens_s_per_user
            previous = pair
    return frontier


class DeploymentTools:
    def __init__(self, request: Request, max_evaluations: int = 2048):
        self.request = request
        self.model = None
        self.gpu = None
        self.plans: dict[str, Plan] = {}
        self.results: dict[str, Result] = {}
        self.trace: list[dict[str, Any]] = []
        self.max_evaluations = max_evaluations

    def load_model_profile(self) -> dict[str, Any]:
        self.model = load_profile(self.request.model_path)
        if self.request.isl + self.request.osl > self.model.max_position_embeddings:
            raise ValueError("ISL + OSL exceeds model context limit")
        return {**asdict(self.model), "parameters_estimate": self.model.parameters, "is_moe": False, "ep": 1}

    def inspect_system(self) -> dict[str, Any]:
        self.gpu = get_gpu(self.request.system)
        return {**asdict(self.gpu), "total_gpus": self.request.total_gpus,
                "effective_mfu": self.request.mfu if self.request.mfu is not None else self.gpu.default_mfu,
                "warning": "effective hardware assumptions, not calibrated serving-backend results"}

    def _simulator(self) -> AnalyticalSimulator:
        if self.model is None or self.gpu is None:
            raise ValueError("Call load_model_profile and inspect_system first")
        return AnalyticalSimulator(self.request, self.model, self.gpu)

    def candidate_plans(self, mode: str, tp_sizes: list[int], pp_sizes: list[int], batch_sizes: list[int],
                        prefill_tp_sizes: list[int], prefill_batch_sizes: list[int], worker_counts: list[int]):
        simulator = self._simulator()
        r = self.request
        if mode not in ("agg", "disagg"):
            raise ValueError("mode must be agg or disagg")
        selections = {
            "tp_sizes": (tp_sizes, r.tp_sizes), "pp_sizes": (pp_sizes, r.pp_sizes),
            "batch_sizes": (batch_sizes, r.batch_sizes), "prefill_tp_sizes": (prefill_tp_sizes, r.tp_sizes),
            "prefill_batch_sizes": (prefill_batch_sizes, r.prefill_batch_sizes),
            "worker_counts": (worker_counts, r.worker_counts),
        }
        for name, (values, allowed) in selections.items():
            if not isinstance(values, list) or any(type(v) is not int or v not in allowed for v in values):
                raise ValueError(f"{name} must be a subset of {list(allowed)}; [] uses all")
        def choose(name):
            values, allowed = selections[name]
            return sorted(set(values or allowed))
        for tp, pp, batch in product(choose("tp_sizes"), choose("pp_sizes"), choose("batch_sizes")):
            d = Worker(tp, pp, batch)
            if mode == "agg":
                if d.gpus > r.total_gpus:
                    continue
                plans = [Plan("agg", d, r.total_gpus // d.gpus)]
            else:
                plans = (
                    Plan("disagg", d, r.total_gpus // unit, p, pw, dw)
                    for ptp, ppp, pb, pw, dw in product(choose("prefill_tp_sizes"), choose("pp_sizes"), choose("prefill_batch_sizes"), choose("worker_counts"), choose("worker_counts"))
                    for p in (Worker(ptp, ppp, pb),)
                    for unit in (p.gpus * pw + d.gpus * dw,)
                    if unit <= r.total_gpus
                )
            for plan in plans:
                try:
                    simulator.validate(plan)
                except ValueError:
                    continue
                yield plan

    def propose_deployments(self, mode: str, tp_sizes: list[int], pp_sizes: list[int], batch_sizes: list[int],
                            prefill_tp_sizes: list[int], prefill_batch_sizes: list[int], worker_counts: list[int], limit: int) -> dict[str, Any]:
        positive_int(limit, "limit")
        if limit > 128:
            raise ValueError("limit cannot exceed 128; refine the search region to avoid truncation")
        plans = list(self.candidate_plans(mode, tp_sizes, pp_sizes, batch_sizes, prefill_tp_sizes, prefill_batch_sizes, worker_counts))
        # Deterministic coarse coverage across a broad region, not hidden performance ranking.
        selected = plans if len(plans) <= limit else [plans[i * (len(plans) - 1) // (limit - 1)] for i in range(limit)] if limit > 1 else plans[:1]
        for plan in selected:
            self.plans[plan.identifier] = plan
        return {"matching_count": len(plans), "returned_count": len(selected), "sampled": len(selected) < len(plans),
                "candidates": [{"id": p.identifier, **asdict(p), "used_gpus": p.used_gpus, "gpus_per_replica": p.gpus_per_replica} for p in selected]}

    def evaluate_deployment(self, candidate_ids: list[str]) -> dict[str, Any]:
        simulator = self._simulator()
        if not isinstance(candidate_ids, list) or not candidate_ids or len(candidate_ids) > 128 or any(not isinstance(v, str) for v in candidate_ids):
            raise ValueError("candidate_ids must contain 1 to 128 strings returned by propose_deployments")
        unknown = set(candidate_ids) - self.plans.keys()
        if unknown:
            raise ValueError(f"Unknown candidate IDs: {sorted(unknown)[:3]}")
        new = set(candidate_ids) - self.results.keys()
        if len(self.results) + len(new) > self.max_evaluations:
            raise ValueError("Evaluation budget exceeded; compare existing results")
        for identifier in candidate_ids:
            if identifier not in self.results:
                self.results[identifier] = simulator.evaluate(self.plans[identifier])
        return {"results": [self.results[i].to_dict() for i in candidate_ids],
                "evaluated_count": len(self.results), "remaining_evaluations": self.max_evaluations - len(self.results)}

    def compare_results(self) -> dict[str, Any]:
        results = list(self.results.values())
        feasible = rank(results)
        closest = sorted((r for r in results if not r.feasible), key=lambda r: (len(r.violations), r.ttft_ms / self.request.ttft + r.tpot_ms / self.request.tpot))[:3]
        return {
            "objective": "maximize output tokens/s per budget GPU subject to TTFT, TPOT and memory constraints",
            "evaluated_count": len(results), "feasible_count": len(feasible),
            "modes_evaluated": sorted({r.plan.mode for r in results}),
            "best": feasible[0].to_dict() if feasible else None,
            "best_by_mode": {mode: next((r.to_dict() for r in feasible if r.plan.mode == mode), None) for mode in ("agg", "disagg")},
            "max_tested_decode_concurrency": max((r.concurrency_capacity for r in feasible), default=0),
            "pareto_ids": [r.identifier for r in pareto_frontier(results)],
            "closest_infeasible": [r.to_dict() for r in closest],
            "warning": "Best among evaluated candidates, not guaranteed global optimum. No queueing/p99 SLA guarantee.",
        }

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in TOOL_NAMES:
            raise ValueError(f"Unknown tool: {name}")
        try:
            output = getattr(self, name)(**arguments)
        except (ValueError, KeyError, TypeError) as exc:
            output = {"error": str(exc)}
        self.trace.append({"tool": name, "arguments": arguments, "observation": output})
        return output


TOOL_NAMES = ("load_model_profile", "inspect_system", "propose_deployments", "evaluate_deployment", "compare_results")


def tool_schemas() -> list[dict[str, Any]]:
    arrays = {name: {"type": "array", "items": {"type": "integer"}, "description": "Subset of config values; [] uses all"}
              for name in ("tp_sizes", "pp_sizes", "batch_sizes", "prefill_tp_sizes", "prefill_batch_sizes", "worker_counts")}
    properties = {
        "load_model_profile": {}, "inspect_system": {}, "compare_results": {},
        "propose_deployments": {"mode": {"type": "string", "enum": ["agg", "disagg"]}, **arrays, "limit": {"type": "integer", "minimum": 1, "maximum": 128}},
        "evaluate_deployment": {"candidate_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 128}},
    }
    descriptions = {
        "load_model_profile": "Load the fixed input model's architecture and parameter estimate. Required before proposing.",
        "inspect_system": "Inspect fixed GPU specs, GPU budget and effective bandwidth/MFU. Required before proposing.",
        "propose_deployments": "Generate valid TP/PP/replica plans in a chosen region. Broad regions are sampled; narrow filters to refine. EP=1, DP is replicas.",
        "evaluate_deployment": "Evaluate returned candidate IDs deterministically: SLA, memory, roofline, stage capacity and bottleneck. Max 128 at once.",
        "compare_results": "Rank feasible evaluated candidates and Pareto tradeoffs; return infeasible violations separately. Call before finishing.",
    }
    return [{"type": "function", "name": name, "description": descriptions[name], "strict": True,
             "parameters": {"type": "object", "properties": properties[name], "required": list(properties[name]), "additionalProperties": False}}
            for name in TOOL_NAMES]
