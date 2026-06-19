@AGENTS.md
@.claude/agent-profile.md

---

# PROJECT STATE (handoff — survives memory compaction)

HackerRank Orchestrate: **Multi-Modal Evidence Review**. Verify damage claims
(`car`/`laptop`/`package`) from images + a chat claim + user history + minimum
evidence requirements; read `dataset/claims.csv` → write `output.csv`; ship an
`evaluation/` folder that scores on `dataset/sample_claims.csv`. Full spec:
[problem_statement.md](problem_statement.md). Rules of engagement: [AGENTS.md](AGENTS.md);
how I operate: [.claude/agent-profile.md](.claude/agent-profile.md). **Challenge ends 2026-06-20 11:00 IST.**

## Operating contract (the director's rules — obey exactly)
- **User is the orchestrator/director; I write the code.** I implement exactly what's proposed, then wait.
- **Approval gate:** write/run NO code until the director approves the step AND all pushback is resolved.
- **No dependency installs without auditing size + getting approval.**
- **Test everything; run tests immediately after building.** On a failing test: report it, STOP, show the test + the failure, update on project state.
- **After a phase's tests pass: notify the director to commit.** I must NEVER commit or push to GitHub.
- **Flag when the `ANTHROPIC_API_KEY` is needed** (a `.env` at P7).
- **Report status + note issues each phase.** Terse + decisive. Surface spec-grounded counterpoints.
- **Mandatory logging:** append a §5.2 entry to `$HOME/hackerrank_orchestrate/log.txt` every turn (onboarding already recorded; never log secrets).

## Locked architecture & decisions
- **Perception (VLM) → Decision (deterministic Python).** One multimodal Claude call per claim OBSERVES (structured findings); pure-Python rules then DECIDE the 14 output fields. Reproducible + unit-testable.
- **Model: `claude-opus-4-8`** (strongest Claude vision). It REJECTS `temperature`/`top_p`/`top_k` → never send them. Reproducibility comes from an **on-disk response cache** (keyed by prompt-version + model + claim text + image bytes), not sampling control. `anthropic` SDK 0.111.0 confirmed to accept `output_config` on `messages.create`.
- **Input role hierarchy (graded):** images = primary truth; conversation defines what to check; `evidence_requirements.csv` gates sufficiency; **`user_history.csv` is risk context ONLY and must NOT flip `claim_status`** (it only adds `user_history_risk` + `manual_review_required`).
- **Severity from the image** (damage extent); forced `unknown` when NEI, `none` when issue is `none`. **Risk flags from history + image quality/authenticity.**
- **Few-shot: format-only, selected by `claim_object` (fixed, NOT similarity-retrieved), with all verdict fields stripped** — example shows shape, no answer to copy.
- **English-only output.** Justifications/reasons in English even for Hinglish claims (golden labels are English, e.g. case_002). Multilingual handled inside the single perception call (`normalized_claim_en`); **NO separate translator model.**
- **Decision layer is deterministic baseline first** (upgrade only if eval warrants).
- **Cost controls:** prompt caching on the invariant prefix; **group test rows by `claim_object`** (identical cached prefix → cache reads); **Anthropic Message Batches API (50% off)** for the offline test run; on-disk cache; bounded concurrency.

## Build plan & status (full suite currently **38 passed**)
- **P0 done (4)** — scaffold, config, entry-point stubs.
- **P1 done (7)** — CSV I/O (14-col contract), image path→id helpers.
- **P2 done (12)** — enums + coercion validators, evidence-requirement lookup, few-shot selector.
- **P3 done (5)** — perception client (base64 vision, JSON-schema output, disk cache, retry).
- **P4 done (10)** — deterministic decision engine + history merge.
- **P5 next (NO gate)** — orchestrator `code/main.py`: parse → perceive → decide → validate → write `output.csv`; load `user_history` (key by `user_id`) and pass `history_row` to `decision.build_output_row`; pass per-object few-shot examples to `PerceptionClient`; group rows by `claim_object`; instrument model-call / token / image / time counters for P6.
- **P6 (NO gate)** — eval harness `code/evaluation/main.py`: run on `sample_claims.csv`, compute per-column accuracy, `claim_status` confusion, risk-flag set F1, exact-row match; emit `code/evaluation/evaluation_report.md` with operational analysis (≈model calls, input/output tokens, #images, projected test-set cost with pricing assumption **Opus 4.8 = $5/$25 per 1M in/out**, latency/runtime, TPM/RPM + batching/caching/retry strategy).
- **P7 (GATE: needs `ANTHROPIC_API_KEY` in `.env`)** — small live slice → tune prompts → full sample eval → run test set → `output.csv`.
- **P8** — packaging (README done; assemble `code.zip`).
- Each phase: code + tests land together, proven by running pytest.

## Module map (`code/`)
- `evidence_review/config.py` — `Settings` (model, dataset paths, `cache_dir`), `get_settings()`, `require_api_key()` (env-only; loads `.env` via python-dotenv, optional).
- `evidence_review/dataio.py` — `INPUT_COLUMNS` (4), **`OUTPUT_COLUMNS` (14, exact spec order)**, `read_csv_rows`, `split_image_paths` (`;`), `image_id`/`image_ids` (stem), `write_output` (QUOTE_ALL + strict 14-col validation, raises on drift).
- `evidence_review/enums.py` — closed vocabularies (CLAIM_OBJECTS, CLAIM_STATUSES, ISSUE_TYPES=12, SEVERITIES=5, RISK_FLAGS=14, per-object OBJECT_PARTS) + coercion: `coerce_claim_status`→`not_enough_information`, `coerce_issue_type`/`coerce_severity`→`unknown`, `coerce_object_part(obj,v)` per-object→`unknown`, `coerce_bool`→`'true'/'false'` (garbage→false), `normalize_risk_flags` (drop invalid, dedupe, drop `none` if others, empty→`none`).
- `evidence_review/requirements.py` — `Requirement` dataclass, `load_requirements`, `for_object` (incl `'all'`: car 6 / laptop 5 / package 6), `find(object, applies_to)` (exact + `'all'` fallback).
- `evidence_review/fewshot.py` — `VERDICT_FIELDS` (10 = output cols minus inputs), `strip_verdict_fields`, `select_examples(rows, claim_object, n)`.
- `evidence_review/perception.py` — `PROMPT_VERSION`, `SYSTEM_PROMPT` (images=truth; in-image text = untrusted→`text_in_image`; multilingual→English; observe-not-verdict), `PERCEPTION_SCHEMA` (claim_interpretation + per_image[] observations + holistic candidates), `encode_image`/`media_type_for`, `cache_key`, `build_system`/`build_user_content`, `PerceptionClient.perceive` (disk cache, **lazy `anthropic` import**, retry, `output_config={"format":{"type":"json_schema","schema":...}}`, no temperature, `max_tokens=4096`).
- `evidence_review/decision.py` — `build_output_row(claim_row, perception, history_row)` + `decide_*` helpers; `history_is_risky`; `assemble_risk_flags`. History never flips status.
- `main.py` / `evaluation/main.py` — entry points (currently P0 stubs; P5/P6 fill them). `tests/` — one test file per module.

## Dataset facts (verified)
- `sample_claims.csv` = 20 labeled rows (all 14 cols); `claims.csv` = 44 input rows (4 cols); 29 sample images, 82 test images.
- `image_paths` are `;`-separated, **relative to `dataset/`** (e.g. `images/test/case_001/img_1.jpg`); image ID = filename stem (`img_1`). Resolve full path = `settings.dataset_dir / rel`.
- Claims can be multilingual (Hinglish, e.g. case_002). Watch for: prompt-injection text inside images (`text_instruction_present`), authenticity (`possible_manipulation`/`non_original_image`), cross-image mismatch (case_002 = two different cars → `claim_mismatch`/`wrong_object`). Risky rows almost always include `manual_review_required`.
- `evidence_requirements.csv` = 11 rows (3 `all`, 3 car, 2 laptop, 3 package); keys: requirement_id, claim_object, applies_to, minimum_image_evidence.
- `user_history.csv` cols: user_id, past_claim_count, accept_claim, manual_review_claim, rejected_claim, last_90_days_claim_count, history_flags, history_summary.

## Environment (IMPORTANT)
- Repo-root `.venv` is **Python 3.14** (homebrew) per `pyvenv.cfg`, with `pytest 9.1.1`, `python-dotenv 1.2.2`, `anthropic 0.111.0` installed in `lib/python3.14/`.
- ⚠️ **`.venv/bin/python` is a STALE 3.13 symlink** (a mistake from running `python3 -m venv` over the user's venv). **Use `.venv/bin/python3.14` for everything.** Tests: `cd code && ../.venv/bin/python3.14 -m pytest`.
- Recommended clean fix (director's shell — sandbox blocks network): `rm -rf .venv && /opt/homebrew/opt/python@3.14/bin/python3.14 -m venv .venv && .venv/bin/pip install anthropic python-dotenv pytest`.
- Tool sandbox blocks network; the director installs deps in their own shell (method "b"). `.gitignore` covers `.venv`, `.env`, `__pycache__`, `.pytest_cache`, `code/.cache/`.

## Known issues / to-do
- **Quality-flag vocab gap (tune in P7):** perception `quality_issues` are free-text; `normalize_risk_flags` keeps only exact-vocab tokens, so image-quality flags (`blurry_image`, etc.) may be under-captured. Fix by tightening the perception prompt to emit exact flag tokens (or enum-constrain `quality_issues`). Boolean-derived flags (manipulation/text/wrong-object/mismatch/history) are captured reliably.
- **`.venv` symlink mismatch** (above) — recommend clean recreate; non-blocking.
- **P0–P4 are flagged ready to commit** (director commits; I never do). Suggested batch msg: `P0–P4: scaffold, IO, enums, perception client, decision engine (38 tests)`.

## Experiments to run later (P7+)
- Few-shot-stripped vs no-few-shot (accuracy on sample).
- Grouped-by-`claim_object` vs ungrouped (cache_read tokens + cost).
