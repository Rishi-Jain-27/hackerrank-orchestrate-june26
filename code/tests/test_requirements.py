"""P2 tests: evidence-requirement loading and lookup."""
from __future__ import annotations

from evidence_review import config, requirements


def _load():
    return requirements.load_requirements(
        config.get_settings().evidence_requirements_csv
    )


def test_load_all_rows():
    reqs = _load()
    assert len(reqs) == 11
    assert all(r.requirement_id and r.minimum_image_evidence for r in reqs)


def test_for_object_includes_all_rules():
    reqs = _load()
    car = requirements.for_object(reqs, "car")
    assert len(car) == 6  # 3 car-specific + 3 'all'
    assert {r.claim_object for r in car} == {"car", "all"}
    assert len(requirements.for_object(reqs, "laptop")) == 5
    assert len(requirements.for_object(reqs, "package")) == 6


def test_find_exact_and_fallback():
    reqs = _load()
    r = requirements.find(reqs, "car", "dent or scratch")
    assert r is not None and r.requirement_id == "REQ_CAR_BODY_PANEL"
    assert requirements.find(reqs, "car", "no such family") is None
