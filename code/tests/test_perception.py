"""P3 tests: perception client — encoding, cache key, message build, mock call.

No network or API key: a fake client is injected. Real images come from the
dataset; the cache dir is redirected to tmp_path.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from evidence_review import config, dataio, perception

# A schema-shaped perception payload used as the fake model response.
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
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload))])


class _FakeMessages:
    def __init__(self, responses):
        self.responses = responses
        self.calls = 0

    def create(self, **kwargs):
        r = self.responses[self.calls]
        self.calls += 1
        if isinstance(r, Exception):
            raise r
        return r


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_media_type_and_encode_real_image():
    s = config.get_settings()
    row = dataio.read_csv_rows(s.sample_claims_csv)[0]
    rel = dataio.split_image_paths(row["image_paths"])[0]
    full = s.dataset_dir / rel
    media_type, b64 = perception.encode_image(full)
    assert media_type == "image/jpeg"
    assert len(b64) > 100


def test_cache_key_is_stable_and_sensitive():
    r1 = {"user_id": "u", "claim_object": "car", "user_claim": "a"}
    r2 = {"user_id": "u", "claim_object": "car", "user_claim": "b"}
    k1 = perception.cache_key(r1, [b"x"], "m")
    assert k1 == perception.cache_key(r1, [b"x"], "m")
    assert k1 != perception.cache_key(r2, [b"x"], "m")  # claim text matters
    assert k1 != perception.cache_key(r1, [b"y"], "m")  # image bytes matter


def test_build_user_content_one_image_block_each():
    images = [("img_1", "image/jpeg", "AAAA"), ("img_2", "image/png", "BBBB")]
    blocks = perception.build_user_content(
        {"claim_object": "car", "user_claim": "scratch on door"}, images
    )
    image_blocks = [b for b in blocks if b.get("type") == "image"]
    assert len(image_blocks) == 2
    assert any(
        "scratch on door" in b.get("text", "")
        for b in blocks
        if b.get("type") == "text"
    )


def test_perceive_parses_and_caches(tmp_path):
    rows = dataio.read_csv_rows(config.get_settings().sample_claims_csv)
    row = rows[0]
    fake = _FakeClient([_fake_response(PAYLOAD), _fake_response(PAYLOAD)])
    s = config.Settings(cache_dir=tmp_path)
    pc = perception.PerceptionClient(settings=s, client=fake)
    out1 = pc.perceive(row)
    assert out1["holistic"]["issue_type_candidate"] == "dent"
    out2 = pc.perceive(row)  # served from cache
    assert out2 == out1
    assert fake.messages.calls == 1  # second call did not hit the API


def test_perceive_retries_then_succeeds(tmp_path):
    rows = dataio.read_csv_rows(config.get_settings().sample_claims_csv)
    row = rows[1]  # case_002: two images, both present in the dataset
    fake = _FakeClient([RuntimeError("transient"), _fake_response(PAYLOAD)])
    s = config.Settings(cache_dir=tmp_path)
    pc = perception.PerceptionClient(settings=s, client=fake, max_retries=3)
    out = pc.perceive(row)
    assert out["holistic"]["evidence_sufficient"] is True
    assert fake.messages.calls == 2  # one failure, one success
