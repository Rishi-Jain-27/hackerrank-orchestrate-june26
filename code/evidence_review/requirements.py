"""Loader and lookup for dataset/evidence_requirements.csv.

The minimum-image-evidence checklist, keyed by (claim_object, issue family).
Pure stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import dataio


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


def load_requirements(path: str | Path) -> list[Requirement]:
    return [
        Requirement(
            requirement_id=r["requirement_id"],
            claim_object=r["claim_object"],
            applies_to=r["applies_to"],
            minimum_image_evidence=r["minimum_image_evidence"],
        )
        for r in dataio.read_csv_rows(path)
    ]


def for_object(reqs: list[Requirement], claim_object: str) -> list[Requirement]:
    """Requirements that apply to a claim_object, including the 'all' rules."""
    return [r for r in reqs if r.claim_object in (claim_object, "all")]


def find(
    reqs: list[Requirement], claim_object: str, applies_to: str
) -> Requirement | None:
    """Exact (claim_object, applies_to) lookup; falls back to an 'all' match."""
    for r in reqs:
        if r.claim_object == claim_object and r.applies_to == applies_to:
            return r
    for r in reqs:
        if r.claim_object == "all" and r.applies_to == applies_to:
            return r
    return None
