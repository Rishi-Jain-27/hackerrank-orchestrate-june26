"""P5 tests: the orchestrator (parse -> perceive -> decide -> validate -> write).

No network or API key: a fake messages client serves canned perception
responses. Real image bytes are loaded from the dataset (sample images); the
cache dir is redirected to tmp_path.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import main
from evidence_review import config, dataio

# A schema-shaped perception payload reused as the fake model response.
PAYLOAD = {
    "claim_interpretation": {
        "issue_family": "dent or scratch",
        "claimed_part": "rear_bumper",
        "language": "en",
        "normalized_claim_en": "dent on the rear bumper",
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
        "reason": "rear bumper visible with a dent",
        "justification": "image shows a dent on the rear bumper",
    },
}


def _fake_response(payload):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=json.dumps(payload))],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
    )


class _FakeMessages:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def create(self, **kwargs):
        r = self.responses[self.calls]
        self.calls += 1
        return r


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


# Two real sample images so bytes load; claim_object spans two groups.
_IMG = "images/sample/case_001/img_1.jpg"


def _write_claims(path):
    rows = [
        ("user_001", _IMG, "the rear bumper has a dent", "car"),
        ("user_009", _IMG, "the laptop screen is cracked", "laptop"),
    ]
    lines = ['"user_id","image_paths","user_claim","claim_object"']
    for r in rows:
        lines.append(",".join(f'"{c}"' for c in r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_group_by_object_preserves_order():
    rows = [
        {"claim_object": "car"},
        {"claim_object": "laptop"},
        {"claim_object": "car"},
    ]
    groups = main.group_by_object(rows)
    assert list(groups) == ["car", "laptop"]
    assert [i for i, _ in groups["car"]] == [0, 2]  # original indices kept


def test_load_history_index_keys_by_user_id():
    idx = main.load_history_index(config.get_settings().user_history_csv)
    assert idx["user_005"]["history_flags"] == "user_history_risk"


def test_run_writes_valid_output_and_counts(tmp_path):
    claims = tmp_path / "claims.csv"
    out = tmp_path / "output.csv"
    _write_claims(claims)
    fake = _FakeClient([_fake_response(PAYLOAD), _fake_response(PAYLOAD)])
    s = config.Settings(cache_dir=tmp_path / "cache")

    stats = main.run(s, claims_csv=claims, output_path=out, client=fake)

    assert stats["written"] == 2
    assert stats["rows"] == 2
    assert stats["groups"] == 2  # car + laptop
    assert stats["api_calls"] == 2
    assert stats["cache_hits"] == 0
    assert stats["images"] == 2
    assert stats["input_tokens"] == 20  # 2 calls x 10
    assert stats["output_tokens"] == 40

    written = dataio.read_csv_rows(out)
    assert list(written[0]) == list(dataio.OUTPUT_COLUMNS)  # exact 14-col order
    # Output preserves input order: car row first, laptop row second.
    assert [r["claim_object"] for r in written] == ["car", "laptop"]
    assert written[0]["user_id"] == "user_001"
    assert written[0]["claim_status"] == "supported"


def test_progress_bar_writes_to_stderr(tmp_path, capsys):
    claims = tmp_path / "claims.csv"
    out = tmp_path / "output.csv"
    _write_claims(claims)
    fake = _FakeClient([_fake_response(PAYLOAD), _fake_response(PAYLOAD)])
    s = config.Settings(cache_dir=tmp_path / "cache")

    stats = main.run(s, claims_csv=claims, output_path=out, client=fake, progress=True)

    captured = capsys.readouterr()
    assert "2/2" in captured.err  # bar reached completion on stderr
    assert "[orchestrate]" in captured.err
    assert captured.out == ""  # nothing leaked to stdout
    assert stats["written"] == 2  # output unaffected


def test_progress_is_silent_by_default(tmp_path, capsys):
    claims = tmp_path / "claims.csv"
    _write_claims(claims)
    fake = _FakeClient([_fake_response(PAYLOAD), _fake_response(PAYLOAD)])
    s = config.Settings(cache_dir=tmp_path / "cache")
    main.run(s, claims_csv=claims, output_path=tmp_path / "o.csv", client=fake)
    assert capsys.readouterr().err == ""  # no bar unless progress=True


def test_run_is_reproducible_via_cache(tmp_path):
    claims = tmp_path / "claims.csv"
    out = tmp_path / "output.csv"
    _write_claims(claims)
    fake = _FakeClient([_fake_response(PAYLOAD), _fake_response(PAYLOAD)])
    s = config.Settings(cache_dir=tmp_path / "cache")

    first = main.run(s, claims_csv=claims, output_path=out, client=fake)
    rows_first = dataio.read_csv_rows(out)

    # Second run: a client with NO responses — every claim must be cache-served.
    fake2 = _FakeClient([])
    second = main.run(s, claims_csv=claims, output_path=out, client=fake2)
    rows_second = dataio.read_csv_rows(out)

    assert first["api_calls"] == 2
    assert second["api_calls"] == 0
    assert second["cache_hits"] == 2
    assert rows_first == rows_second  # identical output across runs
