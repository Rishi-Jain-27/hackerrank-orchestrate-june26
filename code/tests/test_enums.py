"""P2 tests: allowed-value vocabularies and coercion helpers."""
from __future__ import annotations

from evidence_review import enums


def test_vocab_sizes_and_membership():
    assert enums.CLAIM_OBJECTS == frozenset({"car", "laptop", "package"})
    assert len(enums.CLAIM_STATUSES) == 3
    assert len(enums.ISSUE_TYPES) == 12
    assert len(enums.SEVERITIES) == 5
    assert len(enums.RISK_FLAGS) == 14
    assert "front_bumper" in enums.object_parts_for("car")
    assert "screen" in enums.object_parts_for("laptop")
    assert "seal" in enums.object_parts_for("package")
    assert "unknown" in enums.object_parts_for("car")


def test_coerce_claim_status():
    assert enums.coerce_claim_status("supported") == "supported"
    assert enums.coerce_claim_status("maybe") == "not_enough_information"
    assert enums.coerce_claim_status("") == "not_enough_information"


def test_coerce_issue_type_and_severity():
    assert enums.coerce_issue_type("dent") == "dent"
    assert enums.coerce_issue_type("explosion") == "unknown"
    assert enums.coerce_severity("high") == "high"
    assert enums.coerce_severity("catastrophic") == "unknown"


def test_coerce_object_part_is_per_object():
    assert enums.coerce_object_part("car", "door") == "door"
    assert enums.coerce_object_part("car", "screen") == "unknown"  # laptop part
    assert enums.coerce_object_part("laptop", "screen") == "screen"
    assert enums.coerce_object_part("package", "seal") == "seal"


def test_coerce_bool():
    assert enums.coerce_bool(True) == "true"
    assert enums.coerce_bool(False) == "false"
    assert enums.coerce_bool("true") == "true"
    assert enums.coerce_bool("False") == "false"
    assert enums.coerce_bool("garbage") == "false"


def test_normalize_risk_flags():
    assert enums.normalize_risk_flags("none") == "none"
    assert enums.normalize_risk_flags("") == "none"
    assert (
        enums.normalize_risk_flags("blurry_image;claim_mismatch")
        == "blurry_image;claim_mismatch"
    )
    assert enums.normalize_risk_flags("none;blurry_image") == "blurry_image"
    assert enums.normalize_risk_flags("nonsense;blurry_image") == "blurry_image"
    assert enums.normalize_risk_flags("blurry_image;blurry_image") == "blurry_image"
    assert (
        enums.normalize_risk_flags(["wrong_object", "claim_mismatch"])
        == "wrong_object;claim_mismatch"
    )
