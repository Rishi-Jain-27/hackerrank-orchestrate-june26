# Multi-Modal Evidence Review

Verifies damage claims (`car` / `laptop` / `package`) from submitted images, a
short claim conversation, user history, and minimum-evidence requirements, and
writes structured predictions to `output.csv`.

## Design (locked decisions)

- **Images are the primary source of truth.** The conversation defines *what* to
  check; `evidence_requirements.csv` gates *sufficiency*; `user_history.csv` is
  **risk context only** and never overrides clear visual evidence.
- **Perception (VLM) → Decision (deterministic Python).** One multimodal Claude
  call per claim inspects all of its images and returns structured findings;
  pure Python rules then compute the 14 output fields. This keeps verdicts
  reproducible and fully unit-testable.
- **Model:** `claude-opus-4-8` (strongest Claude vision). It does not accept
  `temperature`; reproducibility comes from an **on-disk response cache** keyed
  by `(image bytes + claim + prompt version + model)`, plus structured outputs.
- **Severity** is derived from the **image** (damage extent); **risk flags** are
  derived from **history** + image quality/authenticity signals.
- **Decision rules are deterministic and conservative.** `claim_status` is
  `supported`/`contradicted`/`not_enough_information` from `matches_claim` +
  `evidence_sufficient`; a `contradicted` verdict requires the model's
  **self-rated `assessment_confidence`** to be high/medium (low confidence on a
  non-match → `not_enough_information`). `possible_manipulation` on a submitted
  image forces NEI (untrustworthy evidence). `manual_review_required` is added
  only on a genuine risk flag, low confidence, or NEI — not on every imperfect
  row. History adds `user_history_risk` only; it never flips a verdict.
- **The model is given an explicit vocabulary glossary** with "use X (not Y)
  when…" disambiguation for confusable `issue_type`/`severity` tokens
  (crack vs glass_shatter, stain vs water_damage, dent vs scratch, etc.), so it
  maps observations to the closed enums the way the labels intend. `object_part`
  and `issue_type`/`quality_issues` are additionally hard-constrained by the
  response JSON schema.
- **Few-shot is format-only, selected by `claim_object`** — to teach output
  shape/enum discipline, never to transfer a verdict.
- **Image media types are content-sniffed, not trusted from the extension.**
  The dataset ships images whose `.jpg` extension lies (WebP, PNG, AVIF). We
  detect the real type from magic bytes; AVIF/HEIC (unsupported by the API) are
  transcoded to PNG, and any image over the API's 10 MiB cap is downscaled to a
  JPEG that fits — all at send time via `pillow-heif`/Pillow (the on-disk cache
  key uses the original bytes, so reproducibility is preserved).
- **Cost controls:** prompt caching on the invariant prefix (rows grouped by
  `claim_object`), bounded concurrency, the on-disk cache, and the Anthropic
  **Message Batches API** (50% off) for the offline test run.

## Results (20-row labeled sample)

Scored by `evaluation/main.py` (full report in `evaluation/evaluation_report.md`):

| field | accuracy | | field | accuracy |
|---|---|---|---|---|
| `object_part` | 90% | | `claim_status` | 75% |
| `valid_image` | 90% | | `evidence_standard_met` | 75% |
| `issue_type` | 75% | | `supporting_image_ids` | 65% |
| `severity` | 60% | | `risk_flags` (set F1) | 0.69 |

Exact structured-row match (all 8 structured fields correct): **35%**. `claim_status`
is bounded by the model's run-to-run variance (Opus exposes no `temperature`); the
on-disk cache pins the submitted `output.csv` reproducibly.

## Build plan

| Phase | Deliverable | Status |
|------|-------------|--------|
| P0 | Scaffold, `.venv`, config loader | **done** |
| P1 | CSV I/O (14-col order), image path→id resolver | **done** |
| P2 | Allowed-value enums + validators, requirement lookup, few-shot | **done** |
| P3 | Perception client (base64 vision, JSON schema, retry, cache) | **done** |
| P4 | Deterministic decision engine + history merge | **done** |
| P5 | Orchestrator (`main.py`): parse→perceive→decide→validate→write | **done** |
| P6 | Evaluation harness (`evaluation/main.py`) + `evaluation_report.md` | **done** |
| P7 | Real run: tune on sample → score → predict test → `output.csv` | **done** |
| P8 | Packaging (`code.zip`) | **done** |

Every phase ships with tests and is proven by running them.

## Setup

```sh
# from the repo root
python3 -m venv .venv
.venv/bin/pip install -r code/requirements.txt
export ANTHROPIC_API_KEY=...        # or put it in a local .env (never commit)
```

## Run

```sh
.venv/bin/python code/main.py              # predictions -> output.csv (P5+)
.venv/bin/python code/evaluation/main.py   # score on sample_claims.csv (P6+)
```

## Test

```sh
cd code && ../.venv/bin/python -m pytest
```

69 tests. A handful are integration tests that read the real dataset, so run with
`code/` placed alongside `dataset/` (the repository layout). The pure-logic tests
(enums, dataio, decision, scoring) need no dataset or API key — model calls are
mocked and the perception cache is redirected to a temp dir.

Secrets are read from environment variables only; no API key is needed to run the
test suite.
