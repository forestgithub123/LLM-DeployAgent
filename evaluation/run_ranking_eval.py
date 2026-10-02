#!/usr/bin/env python3
"""Compare analytical best-plan ranking with matched measured benchmark rows."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlsplit
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from deploy_agent.agent import AgentSettings, DeploymentAgent  # noqa: E402
from deploy_agent.domain import Request  # noqa: E402
from deploy_agent.tools import load_profile  # noqa: E402

MODELS_CSV = PROJECT / "evaluation" / "models.csv"
MEASUREMENTS_CSV = PROJECT / "evaluation" / "reference_measurements.csv"
OUTPUT_DIR = PROJECT / "outputs" / "evaluation"
SIGNATURE_FIELDS = ("system", "total_gpus", "precision", "kv_precision", "isl", "osl", "prefix", "ttft", "tpot")
MEASUREMENT_FIELDS = {
    "model_id", "system", "total_gpus", "precision", "kv_precision", "isl", "osl", "prefix",
    "ttft_ms", "tpot_ms", "measured_output_tokens_s_per_gpu", "measured_ttft_ms",
    "measured_tpot_ms", "serving_stack", "stack_version", "measured_mode", "measured_tp",
    "measured_pp", "measured_dp", "source_url",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for pos in range(start, end):
            ranks[order[pos]] = average
        start = end
    return ranks


def spearman_rho(predicted: list[float], measured: list[float]) -> float | None:
    if len(predicted) < 2:
        return None
    x, y = average_ranks(predicted), average_ranks(measured)
    mean_x, mean_y = sum(x) / len(x), sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denom_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    denom_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    return numerator / (denom_x * denom_y) if denom_x and denom_y else None


def pairwise_accuracy(predicted: list[float], measured: list[float]) -> dict[str, int | float | None]:
    concordant = discordant = ties = 0
    for i in range(len(predicted)):
        for j in range(i + 1, len(predicted)):
            pred_sign = (predicted[i] > predicted[j]) - (predicted[i] < predicted[j])
            actual_sign = (measured[i] > measured[j]) - (measured[i] < measured[j])
            if pred_sign == 0 or actual_sign == 0:
                ties += 1
            elif pred_sign == actual_sign:
                concordant += 1
            else:
                discordant += 1
    compared = concordant + discordant
    return {"correct_pairs": concordant, "wrong_pairs": discordant, "tied_pairs": ties,
            "pairwise_accuracy": concordant / compared if compared else None}


def sla_feasibility_metrics(rows: list[dict]) -> dict[str, int | float | None]:
    compared = [row for row in rows if isinstance(row.get("measured_feasible"), bool)]
    correct = sum(row["predicted_feasible"] == row["measured_feasible"] for row in compared)
    true_positive = sum(row["predicted_feasible"] and row["measured_feasible"] for row in compared)
    false_positive = sum(row["predicted_feasible"] and not row["measured_feasible"] for row in compared)
    false_negative = sum(not row["predicted_feasible"] and row["measured_feasible"] for row in compared)
    return {
        "measured_model_count": len(compared),
        "correct_classifications": correct,
        "accuracy": correct / len(compared) if compared else None,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def signature_value(row: dict[str, str], field: str):
    key = "ttft_ms" if field == "ttft" else "tpot_ms" if field == "tpot" else field
    value = row[key]
    if field in ("system", "precision", "kv_precision"):
        return value.strip().lower()
    return float(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic 5-model ranking validation")
    parser.add_argument("--config", default=str(PROJECT / "config.toml"), help="Shared project TOML; [input] is used")
    parser.add_argument("--models", default=str(MODELS_CSV))
    parser.add_argument("--measurements", default=str(MEASUREMENTS_CSV))
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    raw_input = config.get("input", {})
    if not isinstance(raw_input, dict):
        raise ValueError("config [input] must be a TOML table")

    models = read_csv(Path(args.models).resolve())
    expected_model_fields = {"model_id", "huggingface_id", "config_path", "architecture_scope", "config_url"}
    if not models or not expected_model_fields.issubset(models[0]):
        raise ValueError(f"models CSV must have columns: {', '.join(sorted(expected_model_fields))}")
    if len(models) < 5:
        raise ValueError("Ranking validation needs at least five model rows")
    if len({row["model_id"] for row in models}) != len(models):
        raise ValueError("models CSV contains duplicate model_id values")

    missing = []
    resolved_models = []
    for row in models:
        model_config = (PROJECT / row["config_path"]).resolve()
        if not model_config.is_file():
            missing.append(f"{row['model_id']}: {model_config}")
        else:
            resolved_models.append((row, model_config))
    if missing:
        raise ValueError("Download each listed Hugging Face config.json first:\n  " + "\n  ".join(missing))

    predictions = []
    for index, (row, model_config) in enumerate(resolved_models, 1):
        raw_request = {**raw_input, "model_path": str(model_config)}
        request = Request.from_dict(raw_request)
        # Preflight: refuse unsupported hybrid/MoE/MLA/multimodal configs.
        profile = load_profile(str(model_config))
        if request.isl + request.osl > profile.max_position_embeddings:
            raise ValueError(f"{row['model_id']}: ISL+OSL exceeds its context limit")
        print(f"[{index}/{len(resolved_models)}] deterministic exhaustive search: {row['model_id']} ({row['architecture_scope']})")
        run = DeploymentAgent(AgentSettings(mode="baseline")).run(request)
        best = run.best
        predictions.append({
            "model_id": row["model_id"], "huggingface_id": row["huggingface_id"],
            "config_path": str(model_config.relative_to(PROJECT)),
            "predicted_feasible": bool(best),
            "predicted_output_tokens_s_per_gpu": best.tokens_s_per_gpu if best else "",
            "predicted_ttft_ms": best.ttft_ms if best else "",
            "predicted_tpot_ms": best.tpot_ms if best else "",
            "predicted_plan": best.identifier if best else "NO_FEASIBLE_PLAN",
            "evaluated_plans": len(run.results),
            "config_url": row["config_url"],
        })

    measured_rows = read_csv(Path(args.measurements).resolve())
    if measured_rows and not MEASUREMENT_FIELDS.issubset(measured_rows[0]):
        raise ValueError(f"measurements CSV must have columns: {', '.join(sorted(MEASUREMENT_FIELDS))}")
    by_id = {}
    known_ids = {model["model_id"] for model in models}
    for row in measured_rows:
        if row.get("model_id") not in known_ids:
            raise ValueError(f"measurement contains unknown model_id={row.get('model_id')!r}")
        if row["model_id"] in by_id:
            raise ValueError(f"duplicate measurement for model_id={row['model_id']}")
        for field in SIGNATURE_FIELDS:
            expected = raw_input[field]
            actual = signature_value(row, field)
            expected_value = str(expected).strip().lower() if field in ("system", "precision", "kv_precision") else float(expected)
            if actual != expected_value:
                raise ValueError(f"benchmark row {row['model_id']} has mismatched {field}: {row[field if field not in ('ttft','tpot') else field+'_ms']}, expected {expected}")
        if (not row["serving_stack"].strip() or not row["stack_version"].strip()
                or urlsplit(row["source_url"]).scheme not in ("https", "http")
                or not urlsplit(row["source_url"]).netloc):
            raise ValueError(f"benchmark row {row['model_id']} must include stack, version, and source URL")
        for numeric_field in ("measured_output_tokens_s_per_gpu", "measured_ttft_ms", "measured_tpot_ms"):
            try:
                measurement = float(row[numeric_field])
            except ValueError as exc:
                raise ValueError(f"benchmark row {row['model_id']} has invalid {numeric_field}") from exc
            if not math.isfinite(measurement) or measurement <= 0:
                raise ValueError(f"benchmark row {row['model_id']} {numeric_field} must be finite and positive")
        deployment_fields = (row["measured_mode"], row["measured_tp"], row["measured_pp"], row["measured_dp"])
        if any(deployment_fields) and not all(deployment_fields):
            raise ValueError(f"benchmark row {row['model_id']} must fill all measured deployment fields or leave all blank")
        by_id[row["model_id"]] = row

    stacks = {(row["serving_stack"], row["stack_version"]) for row in by_id.values()}
    if len(stacks) > 1:
        raise ValueError("Use one serving_stack and stack_version for all compared measurements")
    source_hosts = {urlsplit(row["source_url"]).netloc.lower() for row in by_id.values()}
    if len(source_hosts) > 1:
        raise ValueError("Use benchmark rows from one source host/dataset; do not mix unrelated performance charts")

    for prediction in predictions:
        measured = by_id.get(prediction["model_id"])
        if measured:
            prediction["measured_output_tokens_s_per_gpu"] = float(measured["measured_output_tokens_s_per_gpu"])
            prediction["measured_ttft_ms"] = float(measured["measured_ttft_ms"])
            prediction["measured_tpot_ms"] = float(measured["measured_tpot_ms"])
            prediction["source_url"] = measured["source_url"]
            prediction["measured_mode"] = measured["measured_mode"]
            prediction["measured_tp"] = measured["measured_tp"]
            prediction["measured_pp"] = measured["measured_pp"]
            prediction["measured_dp"] = measured["measured_dp"]
            prediction["measured_feasible"] = (
                prediction["measured_ttft_ms"] <= float(raw_input["ttft"])
                and prediction["measured_tpot_ms"] <= float(raw_input["tpot"])
            )
        else:
            prediction.update({"measured_output_tokens_s_per_gpu": "", "measured_ttft_ms": "",
                               "measured_tpot_ms": "", "source_url": "", "measured_mode": "",
                               "measured_tp": "", "measured_pp": "", "measured_dp": "", "measured_feasible": ""})

    # Feasibility and throughput ordering are separate questions: first report
    # whether the estimator classifies SLA feasibility correctly, then rank
    # only the models that both the estimator and measurements consider viable.
    feasibility = sla_feasibility_metrics(predictions)

    pred_feasible = [row for row in predictions if row["predicted_feasible"]]
    pred_feasible.sort(key=lambda row: -row["predicted_output_tokens_s_per_gpu"])
    for rank, row in enumerate(pred_feasible, 1):
        row["predicted_rank"] = rank
    matched = [row for row in predictions if row.get("measured_feasible") is True and row["predicted_feasible"]]
    actual_sorted = sorted(matched, key=lambda row: -row["measured_output_tokens_s_per_gpu"])
    for rank, row in enumerate(actual_sorted, 1):
        row["measured_rank"] = rank
    for row in predictions:
        row.setdefault("predicted_rank", "")
        row.setdefault("measured_rank", "")
        row["rank_difference"] = abs(row["predicted_rank"] - row["measured_rank"]) if row["predicted_rank"] and row["measured_rank"] else ""

    metric = pairwise_accuracy(
        [row["predicted_output_tokens_s_per_gpu"] for row in matched],
        [row["measured_output_tokens_s_per_gpu"] for row in matched],
    )
    summary = {
        "model_count": len(predictions), "measurement_count": len(by_id), "matched_and_sla_feasible_count": len(matched),
        "sla_feasibility": feasibility,
        "pairwise_rank": metric,
        "spearman_rho": spearman_rho(
            [row["predicted_output_tokens_s_per_gpu"] for row in matched],
            [row["measured_output_tokens_s_per_gpu"] for row in matched],
        ),
        "top1_agreement": bool(matched and max(matched, key=lambda row: row["predicted_output_tokens_s_per_gpu"])["model_id"]
                                == max(matched, key=lambda row: row["measured_output_tokens_s_per_gpu"])["model_id"]),
        "shared_benchmark_stack": next(iter(stacks), None),
        "validation_status": "matched" if len(matched) >= 5 else "incomplete: need at least five measured models meeting SLA",
        "warning": "Analytical predictions are not serving-stack measurements; conclusions apply only to this matched hardware/workload/stack.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "ranking_comparison.csv"
    columns = list(predictions[0])
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(predictions)
    (OUTPUT_DIR / "ranking_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nPredicted order (analytical best tested plan):")
    for row in pred_feasible:
        actual_rank = row.get("measured_rank", "—")
        print(f"  {row['predicted_rank']}. {row['model_id']}: {row['predicted_output_tokens_s_per_gpu']:.2f} tok/s/GPU; measured rank={actual_rank}")
    print(f"\nMatched measured rows meeting SLA: {len(matched)}/{len(models)}")
    feasibility_accuracy = summary["sla_feasibility"]["accuracy"]
    print(f"SLA feasibility agreement: {feasibility_accuracy if feasibility_accuracy is not None else 'not available'}")
    print(f"Pairwise ranking agreement: {metric['pairwise_accuracy'] if metric['pairwise_accuracy'] is not None else 'not available'}")
    print(f"Spearman rho: {summary['spearman_rho'] if summary['spearman_rho'] is not None else 'not available'}")
    print(f"Validation status: {summary['validation_status']}")
    print(f"Artifacts: {out_csv} and {OUTPUT_DIR / 'ranking_summary.json'}")
    return 0 if len(matched) >= 5 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        raise SystemExit(1)
