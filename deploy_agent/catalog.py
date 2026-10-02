from __future__ import annotations

from .domain import GPUProfile


# Conservative effective hardware profiles; values are assumptions, not live measurements.
GPU_CATALOG: dict[str, GPUProfile] = {
    "H100": GPUProfile("H100", 989, 1978, 80, 3350 * 0.8, 450 * 0.8, 50 * 0.8, 0.35),
    "H20": GPUProfile("H20", 148, 296, 96, 4096 * 0.8, 900 * 0.8 / 2, 50 * 0.8, 0.60),
    "H800": GPUProfile("H800", 989, 1979, 80, 3430 * 0.8, 400 * 0.8 / 2, 50 * 0.8, 0.35),
    "H200": GPUProfile("H200", 989, 1979, 141, 4800 * 0.8, 900 * 0.8 / 2, 50 * 0.8, 0.40),
    "PRO5000": GPUProfile("PRO5000", 274, 536, 72, 1345 * 0.85, 64 * 0.8, 50 * 0.8, 0.50),
}

GPU_ALIASES = {
    "H100_SXM": "H100",
    "H100-SXM": "H100",
    "H200_SXM": "H200",
    "H200-SXM": "H200",
    "H20_SXM": "H20",
    "H800_SXM": "H800",
    "NVIDIA_H200_SXM": "H200",
}


def get_gpu(name: str) -> GPUProfile:
    name = GPU_ALIASES.get(name.upper(), name.upper())
    try:
        profile = GPU_CATALOG[name]
    except KeyError as exc:
        supported = ", ".join(sorted(GPU_CATALOG))
        raise KeyError(f"Unknown GPU {name!r}. Supported GPUs: {supported}") from exc
    return profile
