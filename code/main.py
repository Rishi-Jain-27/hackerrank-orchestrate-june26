"""Terminal entry point (AGENTS.md §6.1).

Runs the evidence-review system on dataset/claims.csv and writes output.csv.
Implemented incrementally — see code/README.md for the build plan. At P0 this
is a wired stub that proves the entry-point contract and configuration load.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the package importable when run as `python code/main.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_review import config  # noqa: E402


def main() -> int:
    settings = config.get_settings()
    print(f"[orchestrate] model   = {settings.model}")
    print(f"[orchestrate] claims  = {settings.claims_csv}")
    print(f"[orchestrate] images  = {settings.images_dir}")
    print("[orchestrate] pipeline not yet implemented (P0 scaffold).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
