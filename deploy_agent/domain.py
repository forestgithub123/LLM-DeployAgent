"""Project-owned input and deployment contracts."""
from dataclasses import asdict, dataclass, field
import math
from typing import Any


@dataclass(frozen=True)
class GPUProfile:
    name: str
    fp16_tflops: float
    fp8_tflops: float
    memory_gb: float
    memory_bandwidth_gbps: float
    nvlink_bandwidth_gbps: float
    rdma_bandwidth_gbps: float
    default_mfu: float


@dataclass(frozen=True)
class ModelProfile:
    name: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 40960
    source: str = "local Hugging Face architecture config"

    @property
    def is_moe(self) -> bool:
        return False

    @property
    def layer_parameters(self) -> int:
        h = self.hidden_size
        q = self.num_attention_heads * self.head_dim
        kv = self.num_kv_heads * self.head_dim
        return 2 * h * q + 2 * h * kv + 3 * h * self.intermediate_size

    @property
    def parameters(self) -> int:
        embeddings = (1 if self.tie_word_embeddings else 2) * self.vocab_size * self.hidden_size
        return self.num_layers * self.layer_parameters + embeddings


def positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive JSON integer")
    return value


@dataclass(frozen=True)
class Request:
    model_path: str
    total_gpus: int = 32
    system: str = "h200_sxm"
    isl: int = 4000
    osl: int = 500
    prefix: int = 500
    ttft: float = 300.0
    tpot: float = 10.0
    precision: str = "fp8"
    kv_precision: str = "fp8"
    tp_sizes: tuple[int, ...] = (1, 2, 4, 8)
    pp_sizes: tuple[int, ...] = (1, 2, 4)
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    prefill_batch_sizes: tuple[int, ...] = (1, 2, 4)
    worker_counts: tuple[int, ...] = (1, 2, 4)
    gpus_per_node: int = 8
    memory_fraction: float = 0.9
    workspace_gb: float = 4.0
    mfu: float | None = None
    scheduler_ms: float = 0.2
    collective_latency_us: float = 5.0
    kv_transfer_gbps: float | None = None
    top_k: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Request":
        fields = set(cls.__dataclass_fields__)
        unknown = set(data) - fields - {"agent", "output_dir", "json"}
        if unknown:
            raise ValueError(f"Unknown config fields: {sorted(unknown)}")
        kwargs = {key: value for key, value in data.items() if key in fields}
        for key in ("tp_sizes", "pp_sizes", "batch_sizes", "prefill_batch_sizes", "worker_counts"):
            if key in kwargs:
                if not isinstance(kwargs[key], list):
                    raise ValueError(f"{key} must be a JSON array")
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("model_path is required")
        if not isinstance(self.system, str) or not self.system.strip():
            raise ValueError("system is required")
        for key in ("total_gpus", "isl", "osl", "gpus_per_node", "top_k"):
            positive_int(getattr(self, key), key)
        if type(self.prefix) is not int or not 0 <= self.prefix < self.isl:
            raise ValueError("prefix must be an integer in [0, isl); a fully cached prompt is not modeled")
        for key in ("tp_sizes", "pp_sizes", "batch_sizes", "prefill_batch_sizes", "worker_counts"):
            values = getattr(self, key)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{key} must contain unique positive integers")
            for value in values:
                positive_int(value, key)
        for key in ("ttft", "tpot", "memory_fraction", "workspace_gb", "scheduler_ms", "collective_latency_us", "mfu", "kv_transfer_gbps"):
            value = getattr(self, key)
            if value is None and key in ("mfu", "kv_transfer_gbps"):
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite non-negative number")
        if self.ttft == 0 or self.tpot == 0 or not 0 < self.memory_fraction <= 1:
            raise ValueError("ttft/tpot must be positive; memory_fraction must be in (0, 1]")
        if self.mfu is not None and not 0 < self.mfu <= 1:
            raise ValueError("mfu must be in (0, 1]")
        if self.kv_transfer_gbps == 0:
            raise ValueError("kv_transfer_gbps must be positive")
        if self.precision not in ("fp8", "bf16") or self.kv_precision not in ("fp8", "bf16"):
            raise ValueError("precision and kv_precision support fp8 or bf16")


@dataclass(frozen=True)
class Worker:
    tp: int
    pp: int
    batch_size: int

    @property
    def gpus(self) -> int:
        return self.tp * self.pp

    @property
    def parallel(self) -> str:
        return f"tp{self.tp}pp{self.pp}ep1"


@dataclass(frozen=True)
class Plan:
    mode: str
    decode: Worker
    replicas: int
    prefill: Worker | None = None
    prefill_workers: int = 0
    decode_workers: int = 1

    @property
    def gpus_per_replica(self) -> int:
        return self.decode.gpus * self.decode_workers + (self.prefill.gpus * self.prefill_workers if self.prefill else 0)

    @property
    def used_gpus(self) -> int:
        return self.replicas * self.gpus_per_replica

    @property
    def identifier(self) -> str:
        d = f"{self.decode.parallel}b{self.decode.batch_size}"
        p = f"{self.prefill.parallel}b{self.prefill.batch_size}" if self.prefill else ""
        return f"{self.mode}:{p}:{self.prefill_workers}P:{d}:{self.decode_workers}D:r{self.replicas}"


@dataclass(frozen=True)
class Result:
    plan: Plan
    feasible: bool
    ttft_ms: float
    tpot_ms: float
    request_latency_ms: float
    total_tokens_s: float
    tokens_s_per_gpu: float
    tokens_s_per_user: float
    requests_s: float
    concurrency_capacity: int
    estimated_active_requests: float
    prefill_ms: float
    kv_transfer_ms: float
    prefill_capacity_requests_s: float
    decode_capacity_requests_s: float
    transfer_capacity_requests_s: float | None
    prefill_memory_gb: float
    decode_memory_gb: float
    bottleneck: str
    violations: tuple[str, ...]
    decode_breakdown_ms: dict[str, float]

    @property
    def identifier(self) -> str:
        return self.plan.identifier

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["id"] = self.identifier
        data["used_gpus"] = self.plan.used_gpus
        data["gpus_per_replica"] = self.plan.gpus_per_replica
        data["dp_replicas"] = self.plan.replicas
        return data


@dataclass
class Run:
    request: Request
    model: ModelProfile
    gpu: GPUProfile
    results: list[Result]
    best: Result | None
    pareto: list[Result]
    best_by_mode: dict[str, Result | None]
    trace: list[dict[str, Any]] = field(default_factory=list)
    agent_mode: str = "scripted"
    stop_reason: str = "completed"
    usage: dict[str, Any] = field(default_factory=dict)
    analysis: str = ""
    report: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": asdict(self.request), "model": asdict(self.model), "gpu": asdict(self.gpu),
            "agent_mode": self.agent_mode, "stop_reason": self.stop_reason,
            "best": self.best.to_dict() if self.best else None,
            "best_by_mode": {key: result.to_dict() if result else None for key, result in self.best_by_mode.items()},
            "pareto": [result.to_dict() for result in self.pareto],
            "results": [result.to_dict() for result in self.results],
            "trace": self.trace, "usage": self.usage, "analysis": self.analysis, "report": self.report,
        }
