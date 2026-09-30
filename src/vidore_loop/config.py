"""Paths and settings. One place; nothing else reads `os.environ` for these.

`Settings` boots keyless: the absence of a key is a legitimate state (CI, tests,
offline scoring) and only a call that would actually reach a model asks for one.
"""

from __future__ import annotations

import os
import subprocess
from functools import cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
# Hugging Face downloads (page images, parquet). Large (~10 GB for every corpus); never committed.
CACHE_DIR = DATA_DIR / "cache"
SPLITS_DIR = DATA_DIR / "splits"
AGENTS_DIR = ROOT / "agents"
RUNS_DIR = ROOT / "runs"
LOOP_DIR = ROOT / "loop"
FRONTEND_DIST = ROOT / "frontend" / "dist"


class Settings(BaseModel):
    """Runtime settings, read once from the environment.

    MLflow is nmp-central-ai's (PLATFORM.md) on :5000; it is never replaced by a
    local store (PLATFORM.md rule 1).
    """

    billing: str = "subscription"
    mlflow_tracking_uri: str = "http://localhost:5000"
    hf_home: Path = CACHE_DIR
    code_sha: str = "unknown"


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@cache
def settings() -> Settings:
    return Settings(
        billing=os.environ.get("BILLING", "subscription"),
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        hf_home=Path(os.environ.get("VIDORE_CACHE_DIR", str(CACHE_DIR))),
        code_sha=_git_sha(),
    )
