"""Terminal entry point (AGENTS.md §6.1): the orchestrator.

Pipeline per claim: parse -> perceive (one multimodal Claude call) -> decide
(deterministic Python) -> validate -> write ``output.csv``. Rows are grouped by
``claim_object`` so each group reuses one cached prompt prefix and one set of
format-only few-shot examples (cost control). Output is emitted back in the
original input order. Run statistics (model calls, cache hits, tokens, images,
wall-clock) are aggregated for the P6 operational report.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the package importable when run as `python code/main.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evidence_review import (  # noqa: E402
    config,
    dataio,
    decision,
    fewshot,
    perception,
)


def load_history_index(path: str | Path) -> dict[str, dict]:
    """Index user_history rows by user_id for O(1) lookup during the run."""
    return {r.get("user_id", ""): r for r in dataio.read_csv_rows(path)}


def group_by_object(rows: list[dict]) -> dict[str, list[tuple[int, dict]]]:
    """Group (original_index, row) pairs by claim_object, preserving order."""
    groups: dict[str, list[tuple[int, dict]]] = {}
    for idx, row in enumerate(rows):
        groups.setdefault(row.get("claim_object", ""), []).append((idx, row))
    return groups


def _empty_stats() -> dict:
    return {
        "claims": 0,
        "images": 0,
        "api_calls": 0,
        "cache_hits": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def run(
    settings: config.Settings | None = None,
    *,
    claims_csv: str | Path | None = None,
    output_path: str | Path | None = None,
    client=None,
    fewshot_n: int = 1,
) -> dict:
    """Run the full pipeline and write output.csv. Returns run statistics.

    `client` is injectable so the orchestrator is testable without the SDK or an
    API key (a fake messages client serves canned perception responses).
    """
    settings = settings or config.get_settings()
    claims_csv = Path(claims_csv) if claims_csv else settings.claims_csv
    output_path = (
        Path(output_path)
        if output_path
        else settings.dataset_dir.parent / "output.csv"
    )

    claim_rows = dataio.read_csv_rows(claims_csv)
    history = load_history_index(settings.user_history_csv)
    sample_rows = dataio.read_csv_rows(settings.sample_claims_csv)
    groups = group_by_object(claim_rows)

    stats = _empty_stats()
    stats["rows"] = len(claim_rows)
    stats["groups"] = len(groups)
    indexed_rows: list[tuple[int, dict]] = []
    started = time.monotonic()

    for claim_object, items in groups.items():
        examples = fewshot.select_examples(sample_rows, claim_object, fewshot_n)
        pc = perception.PerceptionClient(
            settings=settings, client=client, examples=examples
        )
        for idx, claim_row in items:
            perceived = pc.perceive(claim_row)
            history_row = history.get(claim_row.get("user_id", ""))
            out_row = decision.build_output_row(claim_row, perceived, history_row)
            indexed_rows.append((idx, out_row))
        for k in stats:
            if k in pc.stats:
                stats[k] += pc.stats[k]

    stats["seconds"] = round(time.monotonic() - started, 3)

    output_rows = [row for _, row in sorted(indexed_rows, key=lambda x: x[0])]
    dataio.write_output(output_path, output_rows)  # validates 14-col contract
    stats["output_path"] = str(output_path)
    stats["written"] = len(output_rows)
    return stats


def main() -> int:
    settings = config.get_settings()
    print(f"[orchestrate] model  = {settings.model}")
    print(f"[orchestrate] claims = {settings.claims_csv}")
    stats = run(settings)
    print(
        f"[orchestrate] wrote {stats['written']} rows -> {stats['output_path']} "
        f"in {stats['seconds']}s"
    )
    print(
        f"[orchestrate] api_calls={stats['api_calls']} "
        f"cache_hits={stats['cache_hits']} images={stats['images']} "
        f"tokens(in/out)={stats['input_tokens']}/{stats['output_tokens']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
