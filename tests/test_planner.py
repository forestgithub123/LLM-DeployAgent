import unittest
from pathlib import Path
from deploy_agent.agent import AgentSettings, DeploymentAgent

from deploy_agent.domain import Plan, Request, Worker
from deploy_agent.performance_estimator import AnalyticalSimulator
from deploy_agent.tools import DeploymentTools, load_profile
from evaluation.run_ranking_eval import (
    MODELS_CSV, pairwise_accuracy, read_csv, sla_feasibility_metrics, spearman_rho,
)
from evaluation.run_silicon_validation import error_metrics, pairwise_latency_ranking


def request(**changes):
    values = {
        "model_path": str(Path(__file__).parent / "fixtures" / "qwen3-32b-config.json"), "total_gpus": 32, "system": "h200_sxm",
        "isl": 4000, "osl": 500, "prefix": 500, "ttft": 300, "tpot": 10,
    }
    values.update(changes)
    return Request.from_dict(values)


class PlannerTests(unittest.TestCase):
    def test_ranking_metrics_for_perfect_and_reversed_orders(self):
        predicted = [1, 2, 3, 4, 5]
        perfect = pairwise_accuracy(predicted, predicted)
        reversed_order = pairwise_accuracy(predicted, list(reversed(predicted)))
        self.assertEqual(perfect["pairwise_accuracy"], 1.0)
        self.assertEqual(perfect["correct_pairs"], 10)
        self.assertEqual(reversed_order["pairwise_accuracy"], 0.0)
        self.assertAlmostEqual(spearman_rho(predicted, predicted), 1.0)
        self.assertAlmostEqual(spearman_rho(predicted, list(reversed(predicted))), -1.0)

    def test_sla_feasibility_metrics_count_unmatched_rows_as_unknown(self):
        result = sla_feasibility_metrics([
            {"predicted_feasible": True, "measured_feasible": True},
            {"predicted_feasible": True, "measured_feasible": False},
            {"predicted_feasible": False, "measured_feasible": True},
            {"predicted_feasible": False, "measured_feasible": ""},
        ])
        self.assertEqual(result["measured_model_count"], 3)
        self.assertEqual(result["correct_classifications"], 1)
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 1)

    def test_h100_profile_is_available_for_bundled_silicon_rows(self):
        from deploy_agent.catalog import get_gpu
        gpu = get_gpu("h100_sxm")
        self.assertEqual(gpu.name, "H100")
        self.assertEqual(gpu.memory_gb, 80)

    def test_silicon_error_metrics_and_within_group_latency_rank(self):
        metrics = error_metrics([12.0, 18.0], [10.0, 20.0])
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["mae_ms"], 2.0)
        rows = [
            {"tp_size": 1, "pp_size": 1, "dp_size": 1, "batch_size": 1,
             "predicted_request_latency_ms": 10.0, "measured_request_latency_ms": 12.0},
            {"tp_size": 2, "pp_size": 1, "dp_size": 1, "batch_size": 1,
             "predicted_request_latency_ms": 8.0, "measured_request_latency_ms": 9.0},
        ]
        ranking = pairwise_latency_ranking({("same workload",): rows})
        self.assertEqual(ranking["correct_pairs"], 1)
        self.assertEqual(ranking["pairwise_latency_order_agreement"], 1.0)
        self.assertEqual(ranking["top1_lowest_latency_agreement"], 1.0)

    def test_future_throughput_ranking_template_has_five_dense_model_choices(self):
        models = read_csv(MODELS_CSV)
        self.assertEqual(len(models), 5)
        self.assertTrue(all("dense" in model["architecture_scope"] for model in models))
        self.assertEqual(len({model["model_id"] for model in models}), 5)

    def test_bundled_silicon_validation_is_three_base_model_sizes(self):
        data = Path(__file__).resolve().parents[1] / "evaluation" / "aiconfigurator_dense_silicon.csv"
        rows = read_csv(data)
        self.assertEqual(len(rows), 778)
        self.assertEqual(len({row["model_id"] for row in rows}), 4)
        self.assertEqual(len({row["model_config"] for row in rows}), 3)

    def test_model_loader_rejects_multimodal_hybrid_architecture(self):
        hybrid_config = Path(__file__).parent / "fixtures" / "qwen3_5_hybrid.json"
        with self.assertRaisesRegex(ValueError, "dense full-attention"):
            load_profile(str(hybrid_config))

    def test_scripted_mode_runs_tool_workflow_without_llm(self):
        messages = []
        run = DeploymentAgent(
            AgentSettings(mode="scripted", scripted_candidates_per_mode=2),
            progress=messages.append,
        ).run(request())
        self.assertEqual(run.agent_mode, "scripted")
        self.assertEqual(run.usage["api_calls"], 0)
        self.assertEqual({step.get("tool") for step in run.trace} & {
            "load_model_profile", "inspect_system", "propose_deployments",
            "evaluate_deployment", "compare_results",
        }, {"load_model_profile", "inspect_system", "propose_deployments", "evaluate_deployment", "compare_results"})
        self.assertTrue(any("不调用 API" in message for message in messages))

    def test_tools_are_ordered_and_candidate_ids_are_verified(self):
        tools = DeploymentTools(request(), max_evaluations=8)
        self.assertIn("parameters_estimate", tools.call("load_model_profile", {}))
        self.assertEqual(tools.call("inspect_system", {})["name"], "H200")
        proposed = tools.call("propose_deployments", {
            "mode": "agg", "tp_sizes": [8], "pp_sizes": [1], "batch_sizes": [1],
            "prefill_tp_sizes": [], "prefill_batch_sizes": [], "worker_counts": [], "limit": 2,
        })
        self.assertEqual(proposed["returned_count"], 1)
        identifier = proposed["candidates"][0]["id"]
        result = tools.call("evaluate_deployment", {"candidate_ids": [identifier]})
        self.assertEqual(result["evaluated_count"], 1)
        self.assertIn("best", tools.call("compare_results", {}))
        self.assertIn("error", tools.call("evaluate_deployment", {"candidate_ids": ["invented"]}))

    def test_disagg_throughput_is_limited_by_the_slowest_stage(self):
        req = request()
        tools = DeploymentTools(req)
        tools.call("load_model_profile", {})
        tools.call("inspect_system", {})
        plan = Plan("disagg", Worker(8, 1, 8), replicas=2, prefill=Worker(2, 1, 1), prefill_workers=4, decode_workers=1)
        result = tools._simulator().evaluate(plan)
        self.assertAlmostEqual(result.requests_s, min(result.prefill_capacity_requests_s, result.decode_capacity_requests_s, result.transfer_capacity_requests_s))
        self.assertGreater(result.kv_transfer_ms, 0)

    def test_impossible_sla_is_not_marked_feasible(self):
        req = request(ttft=0.01, tpot=0.01)
        tools = DeploymentTools(req)
        tools.call("load_model_profile", {})
        tools.call("inspect_system", {})
        plan = Plan("agg", Worker(8, 1, 1), replicas=4)
        result = tools._simulator().evaluate(plan)
        self.assertFalse(result.feasible)
        self.assertTrue(result.violations)

    def test_invalid_parallelism_is_rejected(self):
        req = request()
        tools = DeploymentTools(req)
        tools.call("load_model_profile", {})
        tools.call("inspect_system", {})
        with self.assertRaises(ValueError):
            tools._simulator().evaluate(Plan("agg", Worker(3, 1, 1), replicas=1))

    def test_model_name_alias_is_not_accepted_without_local_config(self):
        req = request(model_path="Qwen/Qwen3-32B-FP8")
        result = DeploymentTools(req).call("load_model_profile", {})
        self.assertIn("Download the model's Hugging Face config.json", result["error"])


if __name__ == "__main__":
    unittest.main()
