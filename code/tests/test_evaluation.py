"""P6 tests: evaluation harness — scoring math + run/render wiring.

Scoring is tested on hand-built rows with known answers. The end-to-end wiring
(orchestrator -> predictions -> score -> report) is exercised with a fake client
on a 2-row slice of the real sample set (no network, no API key).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from evaluation import main as ev
from evidence_review import config, dataio


def _gold(**over):
    base = {
        "user_id": "user_001",
        "image_paths": "images/sample/case_001/img_1.jpg",
        "user_claim": "dent on the rear bumper",
        "claim_object": "car",
        "evidence_standard_met": "true",
        "evidence_standard_met_reason": "rear bumper visible",
        "risk_flags": "none",
        "issue_type": "dent",
        "object_part": "rear_bumper",
        "claim_status": "supported",
        "claim_status_justification": "image shows the dent",
        "supporting_image_ids": "img_1",
        "valid_image": "true",
        "severity": "medium",
    }
    base.update(over)
    return base


def test_score_predictions_known_values():
    gold = [
        _gold(),
        _gold(
            claim_status="contradicted",
            risk_flags="claim_mismatch;manual_review_required",
            severity="low",
        ),
    ]
    pred = [
        _gold(),  # perfect match
        _gold(
            claim_status="contradicted",
            risk_flags="manual_review_required",  # missing claim_mismatch (FN)
            severity="low",
        ),
    ]
    m = ev.score_predictions(pred, gold)

    assert m["rows"] == 2
    assert m["claim_status_accuracy"] == 1.0
    assert m["claim_status_confusion"]["supported"]["supported"] == 1
    assert m["claim_status_confusion"]["contradicted"]["contradicted"] == 1
    # risk flags: row1 none/none -> nothing; row2 tp=1 (manual_review), fn=1 (mismatch)
    rf = m["risk_flags"]
    assert (rf["tp"], rf["fp"], rf["fn"]) == (1, 0, 1)
    assert rf["precision"] == 1.0
    assert rf["recall"] == 0.5
    assert abs(rf["f1"] - (2 / 3)) < 1e-9
    # row1 fully matches; row2 differs on risk_flags -> exact match 1/2
    assert m["exact_rows"] == 1
    assert m["exact_row_match"] == 0.5


def test_set_fields_compared_order_insensitively():
    gold = [_gold(risk_flags="claim_mismatch;manual_review_required")]
    pred = [_gold(risk_flags="manual_review_required;claim_mismatch")]  # reordered
    m = ev.score_predictions(pred, gold)
    assert m["per_column"]["risk_flags"]["accuracy"] == 1.0
    assert m["exact_row_match"] == 1.0


def test_row_count_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        ev.score_predictions([_gold()], [_gold(), _gold()])


# ---- end-to-end wiring with a fake client -------------------------------------

PAYLOAD = {
    "claim_interpretation": {
        "issue_family": "dent",
        "claimed_part": "rear_bumper",
        "language": "en",
        "normalized_claim_en": "dent on rear bumper",
    },
    "per_image": [
        {
            "image_id": "img_1",
            "shows_object": True,
            "shows_relevant_part": True,
            "issue_visible": True,
            "quality_issues": [],
            "possible_manipulation": False,
            "non_original": False,
            "text_in_image": False,
            "relevant": True,
        }
    ],
    "holistic": {
        "cross_image_consistent": True,
        "supporting_image_ids": ["img_1"],
        "issue_type_candidate": "dent",
        "object_part_candidate": "rear_bumper",
        "severity_estimate": "medium",
        "matches_claim": True,
        "evidence_sufficient": True,
        "reason": "rear bumper visible",
        "justification": "image shows a dent",
    },
}


def _fake_response():
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(PAYLOAD))],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
    )


class _FakeMessages:
    def __init__(self, n):
        self.responses = [_fake_response() for _ in range(n)]
        self.calls = 0

    def create(self, **kwargs):
        r = self.responses[self.calls]
        self.calls += 1
        return r


class _FakeClient:
    def __init__(self, n):
        self.messages = _FakeMessages(n)


def test_run_evaluation_and_render_report(tmp_path):
    # A 2-row slice of the real sample set (keeps real image paths/labels).
    sample_rows = dataio.read_csv_rows(config.get_settings().sample_claims_csv)[:2]
    subset = tmp_path / "sample_subset.csv"
    dataio.write_output(subset, sample_rows)  # full 14-col rows

    s = config.Settings(cache_dir=tmp_path / "cache")
    metrics, run_stats = ev.run_evaluation(
        s,
        sample_csv=subset,
        predictions_path=tmp_path / "pred.csv",
        client=_FakeClient(2),
    )

    assert metrics["rows"] == 2
    total = sum(
        metrics["claim_status_confusion"][g][p]
        for g in metrics["claim_status_confusion"]
        for p in metrics["claim_status_confusion"][g]
    )
    assert total == 2  # confusion matrix accounts for every row
    assert run_stats["api_calls"] == 2
    assert run_stats["input_tokens"] == 2000

    report = ev.render_report(metrics, run_stats, test_rows=44, test_images=82)
    assert "# Evaluation Report" in report
    assert "Operational analysis" in report
    assert "Projected test-set cost" in report
    assert "$" in report


def test_render_report_includes_actual_test_run_telemetry():
    metrics = ev.score_predictions([_gold()], [_gold()])
    run_stats = {
        "api_calls": 20, "cache_hits": 0, "images": 29,
        "input_tokens": 100000, "output_tokens": 10000, "rows": 20, "seconds": 150.0,
    }
    actuals = {
        "api_calls": 44,
        "cache_hits": 0,
        "images": 82,
        "input_tokens": 263284,
        "output_tokens": 24297,
        "seconds": 376.5,
        "written": 44,
    }
    report = ev.render_report(
        metrics, run_stats, test_rows=44, test_images=82, test_actuals=actuals
    )
    assert "Actual test-set run" in report
    assert "Model calls:** 44" in report
    assert "Images processed:** 82" in report
    assert "263,284" in report  # actual input tokens rendered
    assert "376.5 s" in report  # actual runtime rendered
