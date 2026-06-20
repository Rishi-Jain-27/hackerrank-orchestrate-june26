"""Allowed-value vocabularies (problem_statement.md) and coercion helpers.

The model proposes candidate values; these functions force every field onto the
closed enum lists before output, so an off-list value can never reach output.csv.
Pure stdlib.
"""
from __future__ import annotations

from typing import Iterable

CLAIM_OBJECTS: frozenset[str] = frozenset({"car", "laptop", "package"})

CLAIM_STATUSES: frozenset[str] = frozenset(
    {"supported", "contradicted", "not_enough_information"}
)

ISSUE_TYPES: frozenset[str] = frozenset(
    {
        "dent",
        "scratch",
        "crack",
        "glass_shatter",
        "broken_part",
        "missing_part",
        "torn_packaging",
        "crushed_packaging",
        "water_damage",
        "stain",
        "none",
        "unknown",
    }
)

SEVERITIES: frozenset[str] = frozenset({"none", "low", "medium", "high", "unknown"})

RISK_FLAGS: frozenset[str] = frozenset(
    {
        "none",
        "blurry_image",
        "cropped_or_obstructed",
        "low_light_or_glare",
        "wrong_angle",
        "wrong_object",
        "wrong_object_part",
        "damage_not_visible",
        "claim_mismatch",
        "possible_manipulation",
        "non_original_image",
        "text_instruction_present",
        "user_history_risk",
        "manual_review_required",
    }
)

# Image-quality risk flags the perception step may emit in `quality_issues`
# (a subset of RISK_FLAGS). Constraining the model to these exact tokens closes
# the free-text vocab gap so normalize_risk_flags keeps them.
QUALITY_ISSUE_FLAGS: frozenset[str] = frozenset(
    {"blurry_image", "cropped_or_obstructed", "low_light_or_glare", "wrong_angle"}
)

OBJECT_PARTS: dict[str, frozenset[str]] = {
    "car": frozenset(
        {
            "front_bumper",
            "rear_bumper",
            "door",
            "hood",
            "windshield",
            "side_mirror",
            "headlight",
            "taillight",
            "fender",
            "quarter_panel",
            "body",
            "unknown",
        }
    ),
    "laptop": frozenset(
        {
            "screen",
            "keyboard",
            "trackpad",
            "hinge",
            "lid",
            "corner",
            "port",
            "base",
            "body",
            "unknown",
        }
    ),
    "package": frozenset(
        {
            "box",
            "package_corner",
            "package_side",
            "seal",
            "label",
            "contents",
            "item",
            "unknown",
        }
    ),
}


def object_parts_for(claim_object: str) -> frozenset[str]:
    return OBJECT_PARTS.get(claim_object, frozenset({"unknown"}))


def coerce_claim_status(value: str) -> str:
    v = (value or "").strip()
    return v if v in CLAIM_STATUSES else "not_enough_information"


def coerce_issue_type(value: str) -> str:
    v = (value or "").strip()
    return v if v in ISSUE_TYPES else "unknown"


def coerce_severity(value: str) -> str:
    v = (value or "").strip()
    return v if v in SEVERITIES else "unknown"


def coerce_object_part(claim_object: str, value: str) -> str:
    v = (value or "").strip()
    return v if v in object_parts_for(claim_object) else "unknown"


def coerce_bool(value) -> str:
    """Coerce to the string 'true'/'false'. Unknown/garbage -> 'false' (conservative)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return "true" if str(value).strip().lower() in {"true", "t", "yes", "1"} else "false"


def normalize_risk_flags(value: str | Iterable[str]) -> str:
    """Normalize to a ';'-joined string of valid, de-duplicated flags.

    Drops invalid flags; drops 'none' when other flags are present; returns
    'none' when nothing valid remains.
    """
    raw = value.split(";") if isinstance(value, str) else list(value)
    seen: list[str] = []
    for item in raw:
        flag = item.strip()
        if flag in RISK_FLAGS and flag != "none" and flag not in seen:
            seen.append(flag)
    return ";".join(seen) if seen else "none"
