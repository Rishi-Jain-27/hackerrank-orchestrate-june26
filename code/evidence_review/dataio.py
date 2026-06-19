"""CSV I/O and image-path helpers.

Enforces the exact 14-column output order from problem_statement.md and the
semicolon-separated `image_paths` convention. Pure stdlib — no third-party deps.
"""
from __future__ import annotations

import csv
from pathlib import Path

# Input columns in claims.csv (and the leading columns of sample_claims.csv).
INPUT_COLUMNS: tuple[str, ...] = (
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
)

# Required output columns, in the exact order mandated by problem_statement.md.
OUTPUT_COLUMNS: tuple[str, ...] = (
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


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of header-keyed row dicts."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_image_paths(field: str) -> list[str]:
    """Split a semicolon-separated `image_paths` field into individual paths.

    Surrounding whitespace is stripped and empty segments are dropped.
    """
    if not field:
        return []
    return [p.strip() for p in field.split(";") if p.strip()]


def image_id(path: str) -> str:
    """Image ID = filename without extension (e.g. ``.../img_1.jpg`` -> ``img_1``)."""
    return Path(path.strip()).stem


def image_ids(field: str) -> list[str]:
    """Image IDs for a semicolon-separated `image_paths` field, in order."""
    return [image_id(p) for p in split_image_paths(field)]


def write_output(path: str | Path, rows: list[dict[str, str]]) -> None:
    """Write rows to CSV with the exact 14-column order.

    Each row must contain exactly ``OUTPUT_COLUMNS`` — no missing, no extra — so
    schema drift is caught here instead of silently corrupting the submission.
    """
    expected = set(OUTPUT_COLUMNS)
    for i, row in enumerate(rows):
        keys = set(row)
        if keys != expected:
            missing = sorted(expected - keys)
            extra = sorted(keys - expected)
            raise ValueError(
                f"row {i} has wrong columns; missing={missing} extra={extra}"
            )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(OUTPUT_COLUMNS), quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(rows)
