#!/usr/bin/env python3
"""Validate the analytical estimator against bundled AIConfigurator silicon rows.

This is a pointwise TTFT/TPOT and within-workload deployment-latency ranking
validation. AIConfigurator's copied rows do not contain measured output tok/s,
so this script deliberately does not claim a five-model throughput ranking.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from deploy_agent.catalog import get_gpu  # noqa: E402
from deploy_agent.domain import Plan, Request, Worker  # noqa: E402
from deploy_agent.performance_estimator import AnalyticalSimulator  # noqa: E402
from deploy_agent.tools import load_profile  # noqa: E402

DATA = PROJECT / "evaluation" / "aiconfigurator_dense_silicon.csv"
CONFIGS = PROJECT / "evaluation" / "model_configs"
OUTPUT = PROJECT / "outputs" / "evaluation"
SOURCE_FIELDS = {
    "sample_id", "model_id", "model_path", "model_config", "system", "backend",
    "backend_version", "weight_precision", "precision_basis", "mode", "isl", "osl",
    "batch_size", "tp_size", "pp_size", "dp_size", "measured_ttft_ms",
    "measured_tpot_ms", "source_metadata",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def error_metrics(predicted: list[float], measured: list[float]) -> dict[str, float | int | None]:
    if not predicted:
        return {"n": 0, "mae_ms": None, "mape_pct": None, "median_ape_pct": None, "mean_signed_error_ms": None}
    absolute = [abs(p - m) for p, m in zip(predicted, measured)]
    ape = [e / abs(m) * 100 for e, m in zip(absolute, measured) if m != 0]
    return {
        "n": len(predicted),
        "mae_ms": statistics.mean(absolute),
        "mape_pct": statistics.mean(ape) if ape else None,
        "median_ape_pct": statistics.median(ape) if ape else None,
        "mean_signed_error_ms": statistics.mean(p - m for p, m in zip(predicted, measured)),
    }


def pairwise_latency_ranking(groups: dict[tuple, list[dict]]) -> dict[str, int | float | None]:
    correct = incorrect = ties = groups_compared = top1_correct = 0
    for rows in groups.values():
        # The same config may have duplicate source records; collapse it to a
        # median before ranking so repeated records do not inflate pair counts.
        by_config: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            config = (row["tp_size"], row["pp_size"], row["dp_size"], row["batch_size"])
            by_config[config].append(row)
        configs = []
        for config, observations in by_config.items():
            configs.append({
                "config": config,
                "predicted": statistics.median(row["predicted_request_latency_ms"] for row in observations),
                "measured": statistics.median(row["measured_request_latency_ms"] for row in observations),
            })
        if len(configs) < 2:
            continue
        groups_compared += 1
        best_predicted = min(configs, key=lambda item: item["predicted"])
        best_measured = min(configs, key=lambda item: item["measured"])
        top1_correct += best_predicted["config"] == best_measured["config"]
        for i, left in enumerate(configs):
            for right in configs[i + 1:]:
                pred_diff = left["predicted"] - right["predicted"]
                measured_diff = left["measured"] - right["measured"]
                if math.isclose(pred_diff, 0.0, abs_tol=1e-9) or math.isclose(measured_diff, 0.0, abs_tol=1e-9):
                    ties += 1
                elif pred_diff * measured_diff > 0:
                    # Both orderings agree on which config has lower latency.
                    correct += 1
                else:
                    incorrect += 1
    compared = correct + incorrect
    return {
        "workload_groups_compared": groups_compared,
        "correct_pairs": correct,
        "incorrect_pairs": incorrect,
        "tied_pairs_excluded": ties,
        "pairwise_latency_order_agreement": correct / compared if compared else None,
        "top1_lowest_latency_agreement": top1_correct / groups_compared if groups_compared else None,
    }


@lru_cache(maxsize=None)
def profile_for(path: str):
    return load_profile(path)


def predict_row(row: dict[str, str]) -> dict[str, float | int | str]:
    config_path = str((CONFIGS / row["model_config"]).resolve())
    profile = profile_for(config_path)
    tp, pp, dp, batch = (int(row[key]) for key in ("tp_size", "pp_size", "dp_size", "batch_size"))
    used_gpus = tp * pp * dp
    precision = row["weight_precision"]
    system = row["system"]
    request = Request.from_dict({
        "model_path": config_path,
        "total_gpus": used_gpus,
        "system": system,
        "isl": int(row["isl"]),
        "osl": int(row["osl"]),
        "prefix": 0,  # The source file does not specify shared-prefix length.
        "ttft": 1e9, "tpot": 1e9,  # Report point predictions independent of any SLA.
        "precision": precision,
        # AIConfigurator does not publish KV-cache precision in this CSV.
        "kv_precision": "bf16",
        "tp_sizes": [tp], "pp_sizes": [pp], "batch_sizes": [batch],
        "gpus_per_node": 8,
    })
    plan = Plan("agg", Worker(tp, pp, batch), replicas=dp)
    result = AnalyticalSimulator(request, profile, get_gpu(system)).evaluate(plan)
    return {
        "predicted_ttft_ms": result.ttft_ms,
        "predicted_tpot_ms": result.tpot_ms,
        "predicted_request_latency_ms": result.request_latency_ms,
        "predicted_memory_feasible": result.feasible,
        "used_gpus": used_gpus,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA), help="Bundled normalized AIConfigurator silicon rows")
    parser.add_argument("--output-dir", default=str(OUTPUT))
    args = parser.parse_args(argv)
    rows = read_csv(Path(args.data).resolve())
    if not rows or not SOURCE_FIELDS.issubset(rows[0]):
        raise ValueError(f"Silicon data must contain columns: {', '.join(sorted(SOURCE_FIELDS))}")

    predictions: list[dict] = []
    skipped: list[dict[str, str]] = []
    for row in rows:
        try:
            result = predict_row(row)
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            skipped.append({"sample_id": row.get("sample_id", ""), "reason": str(exc)})
            continue
        measured_ttft = float(row["measured_ttft_ms"])
        measured_tpot = float(row["measured_tpot_ms"])
        osl = int(row["osl"])
        predictions.append({
            **row,
            **result,
            "measured_request_latency_ms": measured_ttft + (osl - 1) * measured_tpot,
            "ttft_abs_error_ms": abs(result["predicted_ttft_ms"] - measured_ttft),
            "ttft_ape_pct": abs(result["predicted_ttft_ms"] - measured_ttft) / measured_ttft * 100,
            "tpot_abs_error_ms": abs(result["predicted_tpot_ms"] - measured_tpot),
            "tpot_ape_pct": abs(result["predicted_tpot_ms"] - measured_tpot) / measured_tpot * 100,
        })

    if not predictions:
        raise ValueError("No measurement rows could be evaluated; check model support and config snapshots")

    per_model = {}
    for model_id in sorted({row["model_id"] for row in predictions}):
        selected = [row for row in predictions if row["model_id"] == model_id]
        per_model[model_id] = {
            "rows": len(selected),
            "ttft": error_metrics([r["predicted_ttft_ms"] for r in selected], [float(r["measured_ttft_ms"]) for r in selected]),
            "tpot": error_metrics([r["predicted_tpot_ms"] for r in selected], [float(r["measured_tpot_ms"]) for r in selected]),
        }

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in predictions:
        group_key = tuple(row[field] for field in (
            "model_path", "system", "backend", "backend_version", "weight_precision", "isl", "osl",
        ))
        groups[group_key].append(row)
    summary = {
        "evaluation_scope": "pointwise measured TTFT/TPOT accuracy and within-workload aggregate deployment latency ordering",
        "source": "NVIDIA AIConfigurator silicon_sample.csv (real end-to-end measurements; copied subset documented in README)",
        "input_measurements": len(rows),
        "evaluated_measurements": len(predictions),
        "skipped_measurements": len(skipped),
        "unique_model_ids": len({row["model_id"] for row in predictions}),
        "unique_base_model_sizes": ["Llama 3.1 8B", "Llama 3.1 70B", "Qwen3 32B"],
        "ttft": error_metrics([r["predicted_ttft_ms"] for r in predictions], [float(r["measured_ttft_ms"]) for r in predictions]),
        "tpot": error_metrics([r["predicted_tpot_ms"] for r in predictions], [float(r["measured_tpot_ms"]) for r in predictions]),
        "deployment_configuration_latency_ranking": pairwise_latency_ranking(groups),
        "per_model": per_model,
        "limitations": [
            "The silicon subset contains three distinct dense base model sizes, not five comparable model identities.",
            "The source has measured TTFT/TPOT but no measured output tokens/s/GPU; no model-throughput ranking is claimed.",
            "KV-cache dtype and shared-prefix length are absent from the source; predictions assume BF16 KV and prefix=0.",
            "Rows are heterogeneous across GPU, backend/version, precision and workload; latency ranking is only within exact groups.",
            "Multiple configurations from a sweep are correlated observations, not independent model samples.",
        ],
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "silicon_validation_rows.csv"
    with row_path.open("w", newline="", encoding="utf-8") as handle:
        columns = list(predictions[0])
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(predictions)
    (output_dir / "silicon_validation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    if skipped:
        with (output_dir / "silicon_validation_skipped.csv").open(
            "w", newline="", encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=("sample_id", "reason"))
            writer.writeheader()
            writer.writerows(skipped)

    print(f"Evaluated {len(predictions)}/{len(rows)} real measurements; skipped={len(skipped)}")
    print(f"TTFT MAE={summary['ttft']['mae_ms']:.3f} ms, MAPE={summary['ttft']['mape_pct']:.1f}%")
    print(f"TPOT MAE={summary['tpot']['mae_ms']:.3f} ms, MAPE={summary['tpot']['mape_pct']:.1f}%")
    ranking = summary["deployment_configuration_latency_ranking"]
    print(f"Within-workload latency pairwise agreement={ranking['pairwise_latency_order_agreement']}")
    print(f"Artifacts: {row_path} and {output_dir / 'silicon_validation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
