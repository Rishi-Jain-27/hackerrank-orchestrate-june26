"""Deterministic decision engine: perception findings -> the 14 output fields.

Pure functions, no model calls. The VLM observed; here we DECIDE — reproducibly
and testably. user_history adds risk context only and never changes claim_status
(problem_statement.md: history must not override clear visual evidence). Every
field is forced onto the allowed enums before it leaves this module.
"""
from __future__ import annotations

from . import dataio, enums


def _per_image(perception: dict) -> list[dict]:
    return perception.get("per_image", []) or []


def _holistic(perception: dict) -> dict:
    return perception.get("holistic", {}) or {}


def history_is_risky(history_row: dict | None) -> bool:
    """True if the user's history carries any explicit risk flag."""
    if not history_row:
        return False
    raw = (history_row.get("history_flags", "") or "").strip().lower()
    return bool(raw) and raw != "none"


def decide_valid_image(perception: dict) -> bool:
    """The image set is usable if at least one image is relevant and shows the object."""
    return any(
        img.get("relevant") and img.get("shows_object") for img in _per_image(perception)
    )


def decide_evidence_met(perception: dict) -> bool:
    return bool(_holistic(perception).get("evidence_sufficient")) and decide_valid_image(
        perception
    )


def decide_claim_status(perception: dict) -> str:
    """supported / contradicted / not_enough_information — from the images only."""
    if not decide_evidence_met(perception):
        return "not_enough_information"
    return "supported" if _holistic(perception).get("matches_claim") else "contradicted"


def decide_issue_type(perception: dict) -> str:
    return enums.coerce_issue_type(_holistic(perception).get("issue_type_candidate", ""))


def decide_object_part(perception: dict, claim_object: str) -> str:
    return enums.coerce_object_part(
        claim_object, _holistic(perception).get("object_part_candidate", "")
    )


def decide_severity(perception: dict, claim_status: str, issue_type: str) -> str:
    """Severity from the image; forced unknown when NEI, none when no issue."""
    if claim_status == "not_enough_information":
        return "unknown"
    if issue_type == "none":
        return "none"
    return enums.coerce_severity(_holistic(perception).get("severity_estimate", ""))


def decide_supporting_image_ids(perception: dict, claim_row: dict) -> str:
    """Supporting image IDs, restricted to this claim's actual images; else 'none'."""
    valid = set(dataio.image_ids(claim_row.get("image_paths", "")))
    ids = [
        i for i in (_holistic(perception).get("supporting_image_ids") or []) if i in valid
    ]
    return ";".join(ids) if ids else "none"


def assemble_risk_flags(
    perception: dict, claim_status: str, evidence_met: bool, history_row: dict | None
) -> str:
    """Build the risk_flags string from image quality/authenticity, claim mismatch,
    and history. Adds manual_review_required whenever the case is not a clean
    supported-with-no-concerns."""
    candidates: list[str] = []
    any_issue_visible = False
    for img in _per_image(perception):
        candidates.extend(img.get("quality_issues", []) or [])
        if img.get("possible_manipulation"):
            candidates.append("possible_manipulation")
        if img.get("non_original"):
            candidates.append("non_original_image")
        if img.get("text_in_image"):
            candidates.append("text_instruction_present")
        if not img.get("shows_object"):
            candidates.append("wrong_object")
        elif not img.get("shows_relevant_part"):
            candidates.append("wrong_object_part")
        if img.get("issue_visible"):
            any_issue_visible = True

    if evidence_met and not _holistic(perception).get("matches_claim"):
        candidates.append("claim_mismatch")
    if not any_issue_visible:
        candidates.append("damage_not_visible")
    if history_is_risky(history_row):
        candidates.append("user_history_risk")

    flags = enums.normalize_risk_flags(candidates)
    needs_review = flags != "none" or claim_status != "supported" or not evidence_met
    if needs_review:
        existing = [] if flags == "none" else flags.split(";")
        existing.append("manual_review_required")
        flags = enums.normalize_risk_flags(existing)
    return flags


def build_output_row(
    claim_row: dict, perception: dict, history_row: dict | None = None
) -> dict:
    """Assemble one fully-validated output.csv row (all 14 columns)."""
    claim_object = claim_row.get("claim_object", "")
    valid_image = decide_valid_image(perception)
    evidence_met = decide_evidence_met(perception)
    claim_status = decide_claim_status(perception)
    issue_type = decide_issue_type(perception)
    object_part = decide_object_part(perception, claim_object)
    severity = decide_severity(perception, claim_status, issue_type)
    risk_flags = assemble_risk_flags(perception, claim_status, evidence_met, history_row)
    supporting = decide_supporting_image_ids(perception, claim_row)
    h = _holistic(perception)

    return {
        "user_id": claim_row.get("user_id", ""),
        "image_paths": claim_row.get("image_paths", ""),
        "user_claim": claim_row.get("user_claim", ""),
        "claim_object": claim_object,
        "evidence_standard_met": enums.coerce_bool(evidence_met),
        "evidence_standard_met_reason": (h.get("reason", "") or "").strip(),
        "risk_flags": risk_flags,
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": claim_status,
        "claim_status_justification": (h.get("justification", "") or "").strip(),
        "supporting_image_ids": supporting,
        "valid_image": enums.coerce_bool(valid_image),
        "severity": severity,
    }
