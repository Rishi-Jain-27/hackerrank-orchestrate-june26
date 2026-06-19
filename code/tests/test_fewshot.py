"""P2 tests: format-only few-shot selection (verdict-stripped, by object)."""
from __future__ import annotations

from evidence_review import config, dataio, fewshot


def test_verdict_fields_partition_output_columns():
    inputs = set(dataio.INPUT_COLUMNS)
    verdict = set(fewshot.VERDICT_FIELDS)
    assert inputs.isdisjoint(verdict)
    assert inputs | verdict == set(dataio.OUTPUT_COLUMNS)
    assert len(fewshot.VERDICT_FIELDS) == 10


def test_strip_removes_all_verdict_fields():
    row = {c: "x" for c in dataio.OUTPUT_COLUMNS}
    stripped = fewshot.strip_verdict_fields(row)
    assert set(stripped) == set(dataio.INPUT_COLUMNS)


def test_select_examples_by_object_are_stripped():
    rows = dataio.read_csv_rows(config.get_settings().sample_claims_csv)
    cars = fewshot.select_examples(rows, "car", n=2)
    assert 1 <= len(cars) <= 2
    for ex in cars:
        assert ex["claim_object"] == "car"
        assert not (set(ex) & set(fewshot.VERDICT_FIELDS))
