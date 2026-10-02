#!/usr/bin/env python3
"""Extract supported dense end-to-end rows from AIConfigurator silicon_sample.csv.

The committed evaluation dataset is a reproducible subset of AIConfigurator's
real TTFT/TPOT records. This importer is retained so the subset can be rebuilt
from the upstream CSV without depending on AIConfigurator at evaluation time.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "evaluation" / "aiconfigurator_dense_silicon.csv"
MODELS = {
    "Qwen/Qwen3-32B": ("qwen3-32b.json", "Qwen3-32B"),
    "meta-llama/Llama-3.1-8B-Instruct": ("llama-3.1-8b.json", "Llama-3.1-8B"),
    "nvidia/Llama-3.1-8B-Instruct-FP8": ("llama-3.1-8b.json", "Llama-3.1-8B-FP8"),
    "meta-llama/Llama-3.1-70B": ("llama-3.1-70b.json", "Llama-3.1-70B"),
}
SYSTEMS = {"h100_sxm", "h200_sxm"}
FIELDS = (
    "sample_id", "model_id", "model_path", "model_config", "system", "backend",
    "backend_version", "source_precision", "weight_precision", "precision_basis",
    "mode", "isl", "osl", "batch_size", "tp_size", "pp_size", "dp_size",
    "measured_ttft_ms", "measured_tpot_ms", "source_metadata",
)


def normalized_precision(row: dict[str, str]) -> tuple[str, str] | None:
    raw = (row.get("precision") or "").strip().lower()
    gemm = (row.get("gemm_quant_mode") or "").strip().lower()
    selected = raw or gemm
    if selected == "fp8":
        return "fp8", "reported_fp8"
    if selected in ("bf16", "bfloat16"):
        return "bf16", "reported_bf16"
    if not selected:
        # These model snapshots declare torch_dtype=bfloat16 and the row has no
        # quantization mode. Keep the inference visible in precision_basis.
        return "bf16", "base_checkpoint_dtype_assumption"
    return None


def import_rows(source: Path) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            model_info = MODELS.get(row.get("model_path", ""))
            if not model_info or row.get("system") not in SYSTEMS or row.get("mode") != "agg":
                continue
            if "wip=True" in (row.get("silicon_data_source") or ""):
                continue
            precision = normalized_precision(row)
            if precision is None:
                continue
            try:
                isl, osl = int(row["isl"]), int(row["osl"])
                batch, tp, pp = int(row["batch_size"]), int(row["tp_size"]), int(row["pp_size"])
                dp = int(row["attention_dp_size"] or 1)
                measured_ttft = float(row["silicon_ttft_ms"])
                measured_tpot = float(row["silicon_tpot_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            numbers = (measured_ttft, measured_tpot)
            if min(isl, osl, batch, tp, pp, dp) <= 0 or any(not math.isfinite(v) or v <= 0 for v in numbers):
                continue
            metadata = re.sub(r"/Users/[^/]+/", "<USER_HOME>/", row.get("silicon_data_source", ""))
            output.append({
                "sample_id": row["id"],
                "model_id": model_info[1],
                "model_path": row["model_path"],
                "model_config": model_info[0],
                "system": row["system"],
                "backend": row["backend"],
                "backend_version": row.get("backend_version", ""),
                "source_precision": row.get("precision", ""),
                "weight_precision": precision[0],
                "precision_basis": precision[1],
                "mode": row["mode"],
                "isl": str(isl), "osl": str(osl), "batch_size": str(batch),
                "tp_size": str(tp), "pp_size": str(pp), "dp_size": str(dp),
                "measured_ttft_ms": str(measured_ttft),
                "measured_tpot_ms": str(measured_tpot),
                "source_metadata": metadata,
            })
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="AIConfigurator silicon_sample.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    rows = import_rows(args.source)
    if not rows:
        raise SystemExit("No eligible dense silicon rows found; check the source CSV revision.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Imported {len(rows)} non-WIP dense aggregate measurements -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
