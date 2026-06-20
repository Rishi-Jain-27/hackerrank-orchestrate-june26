"""P4 tests: deterministic decision engine + history merge."""
from __future__ import annotations

from evidence_review import dataio, decision


def _img(**over):
    base = {
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
    base.update(over)
    return base


def _perc(matches=True, evidence=True, issue="dent", part="rear_bumper",
          sev="medium", per_image=None, supporting=("img_1",), cross=True,
          confidence="high"):
    return {
        "claim_interpretation": {
            "issue_family": "dent or scratch",
            "claimed_part": part,
            "language": "en",
            "normalized_claim_en": "dent on rear bumper",
        },
        "per_image": [_img()] if per_image is None else per_image,
        "holistic": {
            "cross_image_consistent": cross,
            "assessment_confidence": confidence,
            "supporting_image_ids": list(supporting),
            "issue_type_candidate": issue,
            "object_part_candidate": part,
            "severity_estimate": sev,
            "matches_claim": matches,
            "evidence_sufficient": evidence,
            "reason": "rear bumper visible; dent present",
            "justification": "image shows the claimed dent",
        },
    }


CLAIM = {
    "user_id": "user_001",
    "image_paths": "images/test/case_x/img_1.jpg",
    "user_claim": "dent on the back",
    "claim_object": "car",
}


def test_supported_clean_case_has_no_risk():
    row = decision.build_output_row(CLAIM, _perc(matches=True, evidence=True))
    assert row["claim_status"] == "supported"
    assert row["evidence_standard_met"] == "true"
    assert row["valid_image"] == "true"
    assert row["risk_flags"] == "none"
    assert row["severity"] == "medium"
    assert row["supporting_image_ids"] == "img_1"
    assert set(row) == set(dataio.OUTPUT_COLUMNS)


def test_contradicted_flags_mismatch_and_manual_review():
    row = decision.build_output_row(CLAIM, _perc(matches=False, evidence=True))
    assert row["claim_status"] == "contradicted"
    assert "claim_mismatch" in row["risk_flags"]
    assert "manual_review_required" in row["risk_flags"]


def test_not_enough_information_forces_unknown_severity():
    row = decision.build_output_row(CLAIM, _perc(evidence=False))
    assert row["claim_status"] == "not_enough_information"
    assert row["severity"] == "unknown"
    assert "manual_review_required" in row["risk_flags"]


def test_issue_none_forces_severity_none():
    row = decision.build_output_row(CLAIM, _perc(issue="none", matches=True, evidence=True))
    assert row["issue_type"] == "none"
    assert row["severity"] == "none"


def test_manipulation_text_and_wrong_object_flags():
    manip = decision.build_output_row(
        CLAIM, _perc(per_image=[_img(possible_manipulation=True)])
    )
    assert "possible_manipulation" in manip["risk_flags"]

    text = decision.build_output_row(CLAIM, _perc(per_image=[_img(text_in_image=True)]))
    assert "text_instruction_present" in text["risk_flags"]

    wrong = decision.build_output_row(
        CLAIM, _perc(per_image=[_img(shows_object=False, relevant=True)])
    )
    assert "wrong_object" in wrong["risk_flags"]
    assert wrong["valid_image"] == "false"  # no usable image
    assert wrong["claim_status"] == "not_enough_information"


def test_cross_image_inconsistency_does_not_force_nei():
    # Reverted P7 fix B: a noisy/secondary inconsistent image must NOT flip a
    # claim the relevant image supports. Status comes from evidence + matches.
    two = [_img(image_id="img_1"), _img(image_id="img_2", relevant=False)]
    row = decision.build_output_row(
        CLAIM, _perc(matches=True, evidence=True, per_image=two, cross=False)
    )
    assert row["claim_status"] == "supported"
    assert "claim_mismatch" not in row["risk_flags"]


def test_clean_supported_has_no_manual_review():
    # manual_review_required is only added on a genuine risk or NEI, not on every
    # non-clean row.
    row = decision.build_output_row(CLAIM, _perc(matches=True, evidence=True))
    assert row["risk_flags"] == "none"


def test_benign_quality_flag_on_usable_image_is_dropped():
    # A quality issue on an otherwise-usable image is noise -> not surfaced, and
    # does not trigger manual review.
    usable = [_img(quality_issues=["low_light_or_glare"])]  # relevant + shows_object
    row = decision.build_output_row(CLAIM, _perc(per_image=usable, matches=True, evidence=True))
    assert row["risk_flags"] == "none"


def test_quality_flag_kept_when_image_unusable():
    # When the evidence is not usable, the quality issue that blocked it is kept.
    blocked = [_img(relevant=False, shows_object=False, quality_issues=["blurry_image"])]
    row = decision.build_output_row(CLAIM, _perc(per_image=blocked, evidence=False))
    assert "blurry_image" in row["risk_flags"]


def test_low_confidence_nonmatch_is_nei_not_contradicted():
    # matches_claim=False but the model is unsure -> can't-verify -> NEI.
    row = decision.build_output_row(
        CLAIM, _perc(matches=False, evidence=True, confidence="low")
    )
    assert row["claim_status"] == "not_enough_information"


def test_confident_nonmatch_is_contradicted():
    row = decision.build_output_row(
        CLAIM, _perc(matches=False, evidence=True, confidence="high")
    )
    assert row["claim_status"] == "contradicted"


def test_low_confidence_adds_manual_review():
    row = decision.build_output_row(
        CLAIM, _perc(matches=True, evidence=True, confidence="low")
    )
    assert row["claim_status"] == "supported"  # confidence does not flip a match
    assert "manual_review_required" in row["risk_flags"]


def test_possible_manipulation_forces_nei():
    manip = [_img(possible_manipulation=True)]
    row = decision.build_output_row(
        CLAIM, _perc(per_image=manip, matches=True, evidence=True)
    )
    assert row["claim_status"] == "not_enough_information"
    assert row["evidence_standard_met"] == "false"
    assert "possible_manipulation" in row["risk_flags"]
    assert "manual_review_required" in row["risk_flags"]


def test_non_original_alone_does_not_sink_a_match():
    # non_original is too noisy to act on -> it must NOT flip a supported claim.
    nonorig = [_img(non_original=True)]
    row = decision.build_output_row(
        CLAIM, _perc(per_image=nonorig, matches=True, evidence=True)
    )
    assert row["claim_status"] == "supported"
    assert "non_original_image" in row["risk_flags"]


def test_history_adds_risk_but_does_not_flip_status():
    risky_history = {"history_flags": "frequent_disputes"}
    row = decision.build_output_row(
        CLAIM, _perc(matches=True, evidence=True), history_row=risky_history
    )
    assert row["claim_status"] == "supported"  # history must NOT override visuals
    assert "user_history_risk" in row["risk_flags"]
    assert "manual_review_required" in row["risk_flags"]


def test_history_does_not_flip_nei_either():
    row = decision.build_output_row(
        CLAIM, _perc(evidence=False), history_row={"history_flags": "x"}
    )
    assert row["claim_status"] == "not_enough_information"


def test_supporting_ids_filtered_to_actual_images():
    row = decision.build_output_row(CLAIM, _perc(supporting=("img_1", "img_99")))
    assert row["supporting_image_ids"] == "img_1"  # img_99 not in this claim
    none_row = decision.build_output_row(CLAIM, _perc(supporting=()))
    assert none_row["supporting_image_ids"] == "none"


def test_off_list_values_are_coerced():
    row = decision.build_output_row(
        CLAIM, _perc(issue="explosion", part="screen")  # screen is a laptop part
    )
    assert row["issue_type"] == "unknown"
    assert row["object_part"] == "unknown"


def test_output_row_is_writable(tmp_path):
    row = decision.build_output_row(CLAIM, _perc())
    out = tmp_path / "output.csv"
    dataio.write_output(out, [row])  # raises if columns are wrong
    assert out.exists()
