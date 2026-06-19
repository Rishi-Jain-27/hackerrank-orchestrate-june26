"""Format-only few-shot selection.

Examples are selected by claim_object (fixed, NOT similarity-retrieved) and have
their verdict-bearing fields stripped, so the model learns input/output shape
without an answer to copy (see code/README.md, few-shot decision). Pure stdlib.
"""
from __future__ import annotations

from . import dataio

# Judgment fields a model could copy from an example = every output column that
# is not an input. Stripping these leaves only the non-verdict (input) context.
VERDICT_FIELDS: tuple[str, ...] = tuple(
    c for c in dataio.OUTPUT_COLUMNS if c not in dataio.INPUT_COLUMNS
)


def strip_verdict_fields(row: dict[str, str]) -> dict[str, str]:
    """Return a copy of a row with all verdict-bearing fields removed."""
    return {k: v for k, v in row.items() if k not in VERDICT_FIELDS}


def select_examples(
    rows: list[dict[str, str]], claim_object: str, n: int = 1
) -> list[dict[str, str]]:
    """Pick up to n verdict-stripped examples for a claim_object (deterministic)."""
    chosen = [r for r in rows if r.get("claim_object") == claim_object][:n]
    return [strip_verdict_fields(r) for r in chosen]
