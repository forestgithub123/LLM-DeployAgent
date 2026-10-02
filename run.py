#!/usr/bin/env python3
"""Single-file launcher: edit config.toml, then run `python run.py`."""
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from deploy_agent.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
