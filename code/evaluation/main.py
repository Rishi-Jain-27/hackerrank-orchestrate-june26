"""Evaluation entry point (AGENTS.md §6.1).

Runs the orchestrator on the labeled `dataset/sample_claims.csv`, scores the
predictions against the expected columns, and writes `evaluation_report.md`
(accuracy + claim_status confusion + risk-flag set F1 + exact-row match +
operational analysis). Pure-Python scoring; the model call goes through the same
injectable, cached `PerceptionClient` the orchestrator uses, so this harness runs
offline against the on-disk cache and is unit-testable with a fake client.
"""
from __future__ import annotations

import sys
from pathlib import Path

# code/evaluation/main.py -> parents[1] == code/  (put the package on the path)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as orchestrator  # noqa: E402  (code/main.py — the P5 orchestrator)
from evidence_review import config, dataio, enums  # noqa: E402

# Structured (closed-vocabulary) verdict fields we score for accuracy / row match.
STRUCTURED_FIELDS: tuple[str, ...] = (
    "evidence_standard_met",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "supporting_image_ids",
    "valid_image",
    "severity",
)
# Compared order-insensitively (they are ';'-separated sets, not ordered strings).
SET_FIELDS: frozenset[str] = frozenset({"risk_flags", "supporting_image_ids"})
# Free-text fields: reported but not accuracy-scored (no single correct string).
TEXT_FIELDS: tuple[str, ...] = (
    "evidence_standard_met_reason",
    "claim_status_justification",
)

# Opus 4.8 pricing assumption (USD per 1M tokens) — see code/README.md.
PRICE_IN_PER_M = 5.0
PRICE_OUT_PER_M = 25.0


def _as_set(value: str) -> frozenset[str]:
    return frozenset(p.strip() for p in (value or "").split(";") if p.strip())


def _as_flagset(value: str) -> frozenset[str]:
    """Risk flags as a set with the 'none' sentinel removed (for F1)."""
    return _as_set(value) - {"none"}


def _field_equal(field: str, a: str, b: str) -> bool:
    if field in SET_FIELDS:
        return _as_set(a) == _as_set(b)
    return (a or "").strip() == (b or "").strip()


def score_predictions(pred_rows: list[dict], gold_rows: list[dict]) -> dict:
    """Compute accuracy / confusion / risk-flag F1 / exact-row metrics.

    Rows are aligned by position (the orchestrator preserves input order).
    """
    if len(pred_rows) != len(gold_rows):
        raise ValueError(
            f"row count mismatch: {len(pred_rows)} predictions vs "
            f"{len(gold_rows)} gold rows"
        )
    n = len(gold_rows)

    per_column = {f: {"correct": 0, "total": n, "accuracy": 0.0} for f in STRUCTURED_FIELDS}
    statuses = sorted(enums.CLAIM_STATUSES)
    confusion = {g: {p: 0 for p in statuses} for g in statuses}
    tp = fp = fn = 0
    exact_rows = 0

    for pred, gold in zip(pred_rows, gold_rows):
        row_ok = True
        for f in STRUCTURED_FIELDS:
            if _field_equal(f, pred.get(f, ""), gold.get(f, "")):
                per_column[f]["correct"] += 1
            else:
                row_ok = False
        if row_ok:
            exact_rows += 1

        g_status = enums.coerce_claim_status(gold.get("claim_status", ""))
        p_status = enums.coerce_claim_status(pred.get("claim_status", ""))
        confusion[g_status][p_status] += 1

        gp = _as_flagset(pred.get("risk_flags", ""))
        gg = _as_flagset(gold.get("risk_flags", ""))
        tp += len(gp & gg)
        fp += len(gp - gg)
        fn += len(gg - gp)

    for f in STRUCTURED_FIELDS:
        per_column[f]["accuracy"] = per_column[f]["correct"] / n if n else 0.0

    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "rows": n,
        "per_column": per_column,
        "claim_status_confusion": confusion,
        "claim_status_accuracy": per_column["claim_status"]["accuracy"],
        "risk_flags": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "exact_row_match": exact_rows / n if n else 0.0,
        "exact_rows": exact_rows,
    }


def run_evaluation(
    settings: config.Settings | None = None,
    *,
    sample_csv: str | Path | None = None,
    predictions_path: str | Path | None = None,
    client=None,
    fewshot_n: int = 1,
    progress: bool = False,
) -> tuple[dict, dict]:
    """Run the orchestrator on the sample set and score it. Returns (metrics, run_stats)."""
    settings = settings or config.get_settings()
    sample_csv = Path(sample_csv) if sample_csv else settings.sample_claims_csv
    predictions_path = (
        Path(predictions_path)
        if predictions_path
        else Path(__file__).resolve().parent / "sample_predictions.csv"
    )
    run_stats = orchestrator.run(
        settings,
        claims_csv=sample_csv,
        output_path=predictions_path,
        client=client,
        fewshot_n=fewshot_n,
        progress=progress,
    )
    pred_rows = dataio.read_csv_rows(predictions_path)
    gold_rows = dataio.read_csv_rows(sample_csv)
    metrics = score_predictions(pred_rows, gold_rows)
    return metrics, run_stats


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render_report(
    metrics: dict,
    run_stats: dict,
    *,
    test_rows: int | None = None,
    test_images: int | None = None,
    model: str = config.DEFAULT_MODEL,
) -> str:
    """Render the markdown evaluation report (accuracy + operational analysis)."""
    api_calls = run_stats.get("api_calls", 0)
    in_tok = run_stats.get("input_tokens", 0)
    out_tok = run_stats.get("output_tokens", 0)
    images = run_stats.get("images", 0)
    rows = run_stats.get("rows", metrics["rows"])

    sample_cost = in_tok / 1e6 * PRICE_IN_PER_M + out_tok / 1e6 * PRICE_OUT_PER_M
    per_call_in = in_tok / api_calls if api_calls else 0
    per_call_out = out_tok / api_calls if api_calls else 0

    L: list[str] = []
    L.append("# Evaluation Report")
    L.append("")
    L.append(
        f"System scored on **{metrics['rows']}** labeled rows of "
        f"`dataset/sample_claims.csv` (model `{model}`)."
    )
    L.append("")

    L.append("## Accuracy (per structured field)")
    L.append("")
    L.append("| field | accuracy | correct / total |")
    L.append("|---|---|---|")
    for f in STRUCTURED_FIELDS:
        c = metrics["per_column"][f]
        L.append(f"| `{f}` | {_pct(c['accuracy'])} | {c['correct']} / {c['total']} |")
    L.append("")
    L.append(
        f"**Exact structured-row match:** {_pct(metrics['exact_row_match'])} "
        f"({metrics['exact_rows']} / {metrics['rows']} rows) — all "
        f"{len(STRUCTURED_FIELDS)} structured fields correct "
        "(`risk_flags`/`supporting_image_ids` compared order-insensitively)."
    )
    L.append("")
    L.append(
        "> Free-text fields `evidence_standard_met_reason` and "
        "`claim_status_justification` are not accuracy-scored (no single correct "
        "string); they are inspected qualitatively."
    )
    L.append("")

    L.append("## claim_status confusion (rows: gold → predicted)")
    L.append("")
    statuses = sorted(enums.CLAIM_STATUSES)
    L.append("| gold ↓ / pred → | " + " | ".join(statuses) + " |")
    L.append("|---|" + "---|" * len(statuses))
    for g in statuses:
        cells = " | ".join(str(metrics["claim_status_confusion"][g][p]) for p in statuses)
        L.append(f"| **{g}** | {cells} |")
    L.append("")
    L.append(f"`claim_status` accuracy: **{_pct(metrics['claim_status_accuracy'])}**")
    L.append("")

    rf = metrics["risk_flags"]
    L.append("## risk_flags (set-based, micro across rows; 'none' excluded)")
    L.append("")
    L.append(
        f"- precision **{_pct(rf['precision'])}**, recall **{_pct(rf['recall'])}**, "
        f"F1 **{rf['f1']:.3f}** (TP={rf['tp']}, FP={rf['fp']}, FN={rf['fn']})"
    )
    L.append("")

    L.append("## Operational analysis")
    L.append("")
    L.append(f"- **Rows processed:** {rows} (1 model call per claim)")
    L.append(f"- **Model calls (this run):** {api_calls}  •  cache hits: {run_stats.get('cache_hits', 0)}")
    L.append(f"- **Images sent:** {images}")
    L.append(f"- **Tokens:** input {in_tok:,} / output {out_tok:,}")
    L.append(f"- **Runtime:** {run_stats.get('seconds', 0)} s")
    L.append(
        f"- **Sample-set cost:** ${sample_cost:.4f} "
        f"(@ ${PRICE_IN_PER_M:.0f}/1M in, ${PRICE_OUT_PER_M:.0f}/1M out)"
    )
    if api_calls == 0:
        L.append(
            "  - _No live API calls this run (served from cache or offline). Token/cost "
            "figures populate on the first cold-cache live run (P7)._"
        )
    L.append("")

    if test_rows:
        proj_in = per_call_in * test_rows
        proj_out = per_call_out * test_rows
        proj_cost = proj_in / 1e6 * PRICE_IN_PER_M + proj_out / 1e6 * PRICE_OUT_PER_M
        L.append("### Projected test-set cost")
        L.append("")
        L.append(
            f"`dataset/claims.csv` = **{test_rows} rows**"
            + (f", **{test_images} images**" if test_images is not None else "")
            + f". At {per_call_in:.0f} in / {per_call_out:.0f} out tokens per claim "
            "(sample average):"
        )
        L.append("")
        L.append(
            f"- est. tokens: input {proj_in:,.0f} / output {proj_out:,.0f}"
        )
        L.append(f"- **est. cost: ${proj_cost:.4f}** (full price)")
        L.append(
            f"- with the **Message Batches API (50% off)**: ~**${proj_cost / 2:.4f}**; "
            "prompt-cache reads on the shared per-`claim_object` prefix cut input cost "
            "further; re-runs are ~$0 (on-disk cache)."
        )
        L.append("")

    L.append("### Throughput, rate limits & resilience")
    L.append("")
    L.append(
        "- **1 call/claim**, bounded concurrency; rows are grouped by `claim_object` "
        "so each group shares one cached prompt prefix (cache reads ≪ full input price)."
    )
    L.append(
        "- **Offline test run** uses the **Message Batches API (50% off)**; latency is "
        "not user-facing, so batch throughput matters more than per-call RPM/TPM."
    )
    L.append(
        "- **Reproducibility:** Opus 4.8 rejects `temperature`; determinism comes from "
        "the on-disk response cache (keyed by prompt-version + model + claim + image "
        "bytes), so re-scoring a cached run is free and byte-identical. A *fresh* "
        "cold run can differ by a row or two (the model is not perfectly "
        "deterministic without temperature control); the cache pins the submitted "
        "`output.csv`."
    )
    L.append(
        "- **Image handling:** media types are content-sniffed (dataset extensions "
        "lie); AVIF/HEIC are transcoded to PNG and any image over the API's 10 MiB "
        "cap is downscaled to a JPEG that fits."
    )
    L.append(
        "- **Resilience:** bounded retries with backoff on transient API/parse errors; "
        "outputs are coerced onto closed enums and validated to the 14-column contract "
        "before write."
    )
    L.append("")
    return "\n".join(L)


def main() -> int:
    settings = config.get_settings()
    print(f"[evaluation] model  = {settings.model}")
    print(f"[evaluation] sample = {settings.sample_claims_csv}")

    metrics, run_stats = run_evaluation(settings, progress=True)

    # Test-set sizing for the cost projection.
    test_claim_rows = dataio.read_csv_rows(settings.claims_csv)
    test_images = sum(
        len(dataio.split_image_paths(r.get("image_paths", ""))) for r in test_claim_rows
    )
    report = render_report(
        metrics,
        run_stats,
        test_rows=len(test_claim_rows),
        test_images=test_images,
        model=settings.model,
    )
    report_path = Path(__file__).resolve().parent / "evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(
        f"[evaluation] claim_status acc={_pct(metrics['claim_status_accuracy'])} "
        f"exact-row={_pct(metrics['exact_row_match'])} "
        f"risk_flag_F1={metrics['risk_flags']['f1']:.3f}"
    )
    print(f"[evaluation] report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
