"""Evaluation entry point (AGENTS.md §6.1).

Runs the system on dataset/sample_claims.csv (labeled) and scores predictions
against the expected columns, then writes evaluation/evaluation_report.md.
Implemented incrementally — see code/README.md. At P0 this is a wired stub.
"""
from __future__ import annotations

import sys
from pathlib import Path

# code/evaluation/main.py -> parents[1] == code/  (put the package on the path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence_review import config  # noqa: E402


def main() -> int:
    settings = config.get_settings()
    print(f"[evaluation] model  = {settings.model}")
    print(f"[evaluation] sample = {settings.sample_claims_csv}")
    print("[evaluation] evaluation not yet implemented (P0 scaffold).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
