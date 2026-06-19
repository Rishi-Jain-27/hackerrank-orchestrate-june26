"""Central configuration: dataset paths, model id, and secret loading.

Secrets are read from environment variables ONLY (AGENTS.md §6.2). A local
`.env` file is loaded if present; it must never be committed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is convenient but must not be required at import time.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - absence/edge cases must not break imports
    pass

# code/evidence_review/config.py -> parents[2] == repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "dataset"

# Perception model. Opus 4.8 has the strongest Claude vision; per the claude-api
# skill it rejects `temperature`, so run-to-run reproducibility comes from the
# on-disk response cache (see code/README.md), not sampling control.
DEFAULT_MODEL = "claude-opus-4-8"


@dataclass(frozen=True)
class Settings:
    """Resolved, read-only configuration for one run."""

    model: str = DEFAULT_MODEL
    dataset_dir: Path = DATASET_DIR
    sample_claims_csv: Path = DATASET_DIR / "sample_claims.csv"
    claims_csv: Path = DATASET_DIR / "claims.csv"
    user_history_csv: Path = DATASET_DIR / "user_history.csv"
    evidence_requirements_csv: Path = DATASET_DIR / "evidence_requirements.csv"
    images_dir: Path = DATASET_DIR / "images"
    cache_dir: Path = REPO_ROOT / "code" / ".cache"

    @property
    def anthropic_api_key(self) -> str | None:
        """Read the key live so tests can monkeypatch the environment."""
        return os.environ.get("ANTHROPIC_API_KEY")

    def require_api_key(self) -> str:
        """Return the API key or raise a clear error. Call this only from code
        that actually performs API requests, never at import time."""
        key = self.anthropic_api_key
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or add it to a local "
                ".env file (never commit secrets)."
            )
        return key


def get_settings() -> Settings:
    return Settings()
