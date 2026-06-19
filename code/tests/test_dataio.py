"""P1 tests: CSV I/O 14-column contract and image-path helpers."""
from __future__ import annotations

import csv

import pytest

from evidence_review import config, dataio


def test_output_columns_exact_order():
    assert dataio.OUTPUT_COLUMNS == (
        "user_id",
        "image_paths",
        "user_claim",
        "claim_object",
        "evidence_standard_met",
        "evidence_standard_met_reason",
        "risk_flags",
        "issue_type",
        "object_part",
        "claim_status",
        "claim_status_justification",
        "supporting_image_ids",
        "valid_image",
        "severity",
    )
    assert len(dataio.OUTPUT_COLUMNS) == 14


def test_split_image_paths():
    assert dataio.split_image_paths("a/img_1.jpg") == ["a/img_1.jpg"]
    assert dataio.split_image_paths("a/img_1.jpg;a/img_2.jpg") == [
        "a/img_1.jpg",
        "a/img_2.jpg",
    ]
    assert dataio.split_image_paths(" a/img_1.jpg ; a/img_2.jpg ") == [
        "a/img_1.jpg",
        "a/img_2.jpg",
    ]
    assert dataio.split_image_paths("") == []
    assert dataio.split_image_paths("a;;b") == ["a", "b"]


def test_image_id_and_ids():
    assert dataio.image_id("images/test/case_001/img_1.jpg") == "img_1"
    assert dataio.image_id("img_2.png") == "img_2"
    assert dataio.image_ids(
        "images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg"
    ) == ["img_1", "img_2"]


def test_read_real_claims_dataset():
    claims = dataio.read_csv_rows(config.get_settings().claims_csv)
    assert len(claims) == 44
    assert set(claims[0]) == set(dataio.INPUT_COLUMNS)


def test_read_real_sample_dataset_has_all_columns():
    rows = dataio.read_csv_rows(config.get_settings().sample_claims_csv)
    assert len(rows) == 20
    assert set(rows[0]) == set(dataio.OUTPUT_COLUMNS)


def test_write_output_roundtrip(tmp_path):
    row = {c: "x" for c in dataio.OUTPUT_COLUMNS}
    row["user_id"] = "user_001"
    row["claim_status"] = "supported"
    out = tmp_path / "output.csv"
    dataio.write_output(out, [row])
    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = next(reader)
    assert header == list(dataio.OUTPUT_COLUMNS)
    assert dict(zip(header, data))["user_id"] == "user_001"
    assert dict(zip(header, data))["claim_status"] == "supported"


def test_write_output_rejects_bad_columns(tmp_path):
    out = tmp_path / "bad.csv"
    with pytest.raises(ValueError):
        dataio.write_output(out, [{"user_id": "u"}])  # missing columns
    with pytest.raises(ValueError):
        bad = {c: "x" for c in dataio.OUTPUT_COLUMNS}
        bad["extra_col"] = "y"
        dataio.write_output(out, [bad])
