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
- **Few-shot is format-only, selected by `claim_object`** — to teach output
  shape/enum discipline, never to transfer a verdict.
- **Cost controls:** prompt caching on the invariant prefix (rows grouped by
  `claim_object`), bounded concurrency, the on-disk cache, and the Anthropic
  **Message Batches API** (50% off) for the offline test run.

## Build plan

| Phase | Deliverable | Status |
|------|-------------|--------|
| P0 | Scaffold, `.venv`, config loader | **done** |
| P1 | CSV I/O (14-col order), image path→id resolver | **done** |
| P2 | Allowed-value enums + validators, requirement lookup, few-shot | **done** |
| P3 | Perception client (base64 vision, JSON schema, retry, cache) | todo |
| P4 | Deterministic decision engine + history merge | todo |
| P5 | Orchestrator (`main.py`): parse→perceive→decide→validate→write | todo |
| P6 | Evaluation harness (`evaluation/main.py`) + `evaluation_report.md` | todo |
| P7 | Real run: tune on sample → score → predict test → `output.csv` | todo |
| P8 | Packaging | todo |

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

Secrets are read from environment variables only.
