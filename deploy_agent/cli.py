"""One config-driven entry point for the deployment-planning application."""
import argparse
import json
from pathlib import Path
import sys
import time
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    try:
        import tomli as tomllib
    except ModuleNotFoundError:
        tomllib = None

from .agent import AgentSettings, DeploymentAgent
from .domain import Request
from .tools import load_profile


def main(argv=None) -> int:
    default_config = Path(__file__).resolve().parents[1] / "config.toml"
    parser = argparse.ArgumentParser(description="LLM-DeployAgent: config-driven deployment planning")
    parser.add_argument("--config", default=str(default_config), help="TOML config (default: project config.toml)")
    parser.add_argument("--mode", choices=["scripted", "live"], help="Temporarily override run.mode")
    parser.add_argument("--json", action="store_true", help="Print machine-readable results")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    try:
        config_path = Path(args.config).resolve()
        if tomllib is None:
            raise RuntimeError("TOML support for Python 3.10 requires: python3 -m pip install tomli")
        data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Config must contain TOML tables")
        allowed_sections = {"run", "input", "agent"}
        if set(data) - allowed_sections:
            raise ValueError(f"Unknown top-level config sections: {sorted(set(data) - allowed_sections)}")
        run_data = data.get("run", {})
        input_data = data.get("input", {})
        settings_data = data.get("agent", {})
        for name, section in (("run", run_data), ("input", input_data), ("agent", settings_data)):
            if not isinstance(section, dict):
                raise ValueError(f"[{name}] must be a TOML table")
        request = Request.from_dict(input_data)
        # Local model paths resolve relative to the config, not the caller's cwd.
        local_path = config_path.parent / request.model_path
        if local_path.exists():
            from dataclasses import replace
            request = replace(request, model_path=str(local_path.resolve()))
        # Validate the local architecture before making any potentially billable
        # live-mode API call; the agent tool loop will load it again for its trace.
        model_profile = load_profile(request.model_path)
        if request.isl + request.osl > model_profile.max_position_embeddings:
            raise ValueError("ISL + OSL exceeds model context limit")
        mode = args.mode or run_data.get("mode", "scripted")
        if mode not in ("scripted", "live"):
            raise ValueError("run.mode must be 'scripted' or 'live'")
        settings_data = {**settings_data, "mode": mode}
        settings = AgentSettings(**settings_data)
        # Validate output settings before running any potentially billable calls.
        output_dir = run_data.get("output_dir", "outputs/latest")
        if not isinstance(output_dir, str) or not output_dir:
            raise ValueError("output_dir must be a non-empty string")
        output = Path(output_dir)
        if not output.is_absolute():
            output = config_path.parent / output
        if output.exists() and not output.is_dir():
            raise ValueError("output_dir must be a directory")
        progress = print if mode == "scripted" and not (args.json or run_data.get("json", False)) else (lambda _: None)
        if mode == "scripted" and not (args.json or run_data.get("json", False)):
            print("LLM-DeployAgent · scripted transparent run (no API call)")
        run = DeploymentAgent(settings, progress=progress).run(request)
        elapsed = time.perf_counter() - started
        payload = {**run.to_dict(), "elapsed_seconds": elapsed}
        output.mkdir(parents=True, exist_ok=True)
        (output / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / "report.txt").write_text(run.report + "\n", encoding="utf-8")
        (output / "agent_trace.json").write_text(json.dumps(run.trace, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.json or run_data.get("json", False):
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(run.report)
            print(f"\nCompleted in {elapsed:.2f}s. Artifacts: {output.resolve()}")
        return 0 if run.best is not None and run.stop_reason == "completed" else 2
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Keep provider/network failures readable. The full SDK traceback is
        # rarely useful to a user running the one-file launcher.
        status = getattr(exc, "status_code", None)
        if status == 401:
            print("Error: LLM authentication failed (HTTP 401). Check the key and provider/base_url; an OpenRouter sk-or-v1 key must use https://openrouter.ai/api/v1.", file=sys.stderr)
        else:
            print(f"LLM provider error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
