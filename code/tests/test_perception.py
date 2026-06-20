"""P3 tests: perception client — encoding, cache key, message build, mock call.

No network or API key: a fake client is injected. Real images come from the
dataset; the cache dir is redirected to tmp_path.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from evidence_review import config, dataio, enums, perception

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
    # Media type is content-sniffed (dataset extensions lie), and must be one the
    # Anthropic API accepts.
    assert media_type in {"image/jpeg", "image/png", "image/gif", "image/webp"}
    assert media_type == perception.detect_media_type(full.read_bytes(), full)
    assert len(b64) > 100


def test_sniff_media_type_from_magic_bytes():
    assert perception.sniff_media_type(b"\xff\xd8\xff\xe0junk") == "image/jpeg"
    assert perception.sniff_media_type(b"\x89PNG\r\n\x1a\nxx") == "image/png"
    assert perception.sniff_media_type(b"GIF89a....") == "image/gif"
    assert perception.sniff_media_type(b"RIFF????WEBPVP8 ") == "image/webp"
    assert perception.sniff_media_type(b"\x00\x00\x00\x1cftypavif\x00\x00\x00\x00") == "image/avif"
    assert perception.sniff_media_type(b"not-an-image") is None


def test_avif_dataset_image_is_converted_to_png():
    """The dataset ships AVIF files as .jpg; prepare_image must transcode to PNG."""
    import pytest

    pytest.importorskip("pillow_heif")
    s = config.get_settings()
    avif = s.dataset_dir / "images/test/case_001/img_1.jpg"  # actually AVIF
    raw = avif.read_bytes()
    assert perception.detect_media_type(raw, avif) == "image/avif"  # unsupported as-is

    media_type, b64 = perception.prepare_image(raw, avif)
    assert media_type == "image/png"  # transcoded to a supported type
    import base64 as _b64

    assert _b64.standard_b64decode(b64)[:8] == b"\x89PNG\r\n\x1a\n"  # valid PNG bytes


def test_b64_len_matches_real_encoding():
    import base64 as _b64

    for raw in (b"", b"a", b"ab", b"abc", b"abcd", b"x" * 1000):
        assert perception._b64_len(raw) == len(_b64.standard_b64encode(raw))


def test_oversized_image_is_downscaled_to_jpeg(monkeypatch):
    import io
    import os

    import pytest

    pytest.importorskip("PIL")
    from PIL import Image

    # Random pixels barely compress -> a genuinely large encode that must shrink.
    big = Image.frombytes("RGB", (2000, 2000), os.urandom(2000 * 2000 * 3))
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    data = buf.getvalue()

    monkeypatch.setattr(perception, "MAX_IMAGE_B64_BYTES", 1_000_000)
    out = perception._shrink_to_limit(data)
    assert perception._b64_len(out) <= 1_000_000
    assert out[:3] == b"\xff\xd8\xff"  # re-encoded as JPEG


def test_prepare_image_passes_through_supported_formats():
    s = config.get_settings()
    webp = s.dataset_dir / "images/sample/case_002/img_2.jpg"  # actually WebP (supported)
    media_type, _ = perception.prepare_image(webp.read_bytes(), webp)
    assert media_type == "image/webp"  # no conversion


def test_detect_media_type_prefers_content_over_extension():
    # Content wins when the extension lies (the case_002 bug: WebP saved as .jpg).
    assert perception.detect_media_type(b"RIFF????WEBP", "photo.jpg") == "image/webp"
    # Falls back to the extension when content is unrecognizable.
    assert perception.detect_media_type(b"\x00\x01\x02", "photo.png") == "image/png"


def test_detect_media_type_on_real_mislabeled_dataset_image():
    # dataset/images/sample/case_002/img_2.jpg is actually a WebP file.
    s = config.get_settings()
    full = s.dataset_dir / "images/sample/case_002/img_2.jpg"
    media_type, _ = perception.encode_image(full)
    assert media_type == "image/webp"
    # And the genuinely-jpeg sibling is still detected as jpeg.
    jpeg = s.dataset_dir / "images/sample/case_002/img_1.jpg"
    assert perception.encode_image(jpeg)[0] == "image/jpeg"


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


def test_build_user_content_injects_parts_and_standards():
    blocks = perception.build_user_content(
        {"claim_object": "car", "user_claim": "dent"},
        [("img_1", "image/jpeg", "AAAA")],
        allowed_parts=["rear_bumper", "door", "unknown"],
        requirements_text=["The claimed car panel should be visible."],
    )
    text = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
    assert "rear_bumper" in text and "door" in text  # allowed object_part tokens
    assert "The claimed car panel should be visible." in text  # injected standard


def test_build_schema_constrains_object_part_and_quality_to_enums():
    schema = perception.build_schema("laptop")
    part = schema["properties"]["holistic"]["properties"]["object_part_candidate"]
    assert set(part["enum"]) == set(enums.object_parts_for("laptop"))
    quality = schema["properties"]["per_image"]["items"]["properties"]["quality_issues"]
    assert set(quality["items"]["enum"]) == set(enums.QUALITY_ISSUE_FLAGS)
    # The module-level schema is untouched (per-object copy).
    assert perception.PERCEPTION_SCHEMA["properties"]["holistic"]["properties"][
        "object_part_candidate"
    ] == {"type": "string"}


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


def test_system_prompt_has_vocabulary_disambiguation():
    sysp = perception.build_system()
    # Confusable-pair guidance must be present (the p7-5 glossary).
    for token in ("glass_shatter", "crack", "stain", "water_damage", "torn_packaging"):
        assert token in sysp
    assert "NOT `glass_shatter`" in sysp  # explicit use-X-not-Y boundary
    assert "still `medium`" in sysp        # severity high-vs-medium boundary


def test_schema_includes_assessment_confidence_enum():
    h = perception.PERCEPTION_SCHEMA["properties"]["holistic"]
    assert "assessment_confidence" in h["required"]
    assert h["properties"]["assessment_confidence"]["enum"] == ["high", "medium", "low"]


def test_perceive_retries_on_empty_per_image(tmp_path):
    rows = dataio.read_csv_rows(config.get_settings().sample_claims_csv)
    row = rows[0]
    empty = {**PAYLOAD, "per_image": []}  # degenerate first response
    fake = _FakeClient([_fake_response(empty), _fake_response(PAYLOAD)])
    s = config.Settings(cache_dir=tmp_path)
    pc = perception.PerceptionClient(settings=s, client=fake, max_retries=3)
    out = pc.perceive(row)
    assert out["per_image"]  # recovered a non-empty response
    assert fake.messages.calls == 2  # first empty response was retried


def test_perceive_retries_then_succeeds(tmp_path):
    rows = dataio.read_csv_rows(config.get_settings().sample_claims_csv)
    row = rows[1]  # case_002: two images, both present in the dataset
    fake = _FakeClient([RuntimeError("transient"), _fake_response(PAYLOAD)])
    s = config.Settings(cache_dir=tmp_path)
    pc = perception.PerceptionClient(settings=s, client=fake, max_retries=3)
    out = pc.perceive(row)
    assert out["holistic"]["evidence_sufficient"] is True
    assert fake.messages.calls == 2  # one failure, one success
