"""P0 tests: configuration loads, paths resolve, and secret handling is safe."""
from __future__ import annotations

import pytest

from evidence_review import config


def test_default_model_is_opus_4_8():
    assert config.get_settings().model == "claude-opus-4-8"


def test_dataset_paths_resolve_into_repo():
    s = config.get_settings()
    assert s.claims_csv.name == "claims.csv"
    assert s.sample_claims_csv.name == "sample_claims.csv"
    assert s.evidence_requirements_csv.name == "evidence_requirements.csv"
    assert s.user_history_csv.name == "user_history.csv"
    # The dataset directory actually exists in this repo.
    assert s.dataset_dir.is_dir()


def test_require_api_key_raises_when_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = config.get_settings()
    assert s.anthropic_api_key is None
    with pytest.raises(RuntimeError):
        s.require_api_key()


def test_require_api_key_returns_when_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
    assert config.get_settings().require_api_key() == "sk-test-123"
