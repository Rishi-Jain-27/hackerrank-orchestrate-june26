# Evaluation Report

System scored on **20** labeled rows of `dataset/sample_claims.csv` (model `claude-opus-4-8`).

## Accuracy (per structured field)

| field | accuracy | correct / total |
|---|---|---|
| `evidence_standard_met` | 75.0% | 15 / 20 |
| `risk_flags` | 55.0% | 11 / 20 |
| `issue_type` | 75.0% | 15 / 20 |
| `object_part` | 90.0% | 18 / 20 |
| `claim_status` | 75.0% | 15 / 20 |
| `supporting_image_ids` | 65.0% | 13 / 20 |
| `valid_image` | 90.0% | 18 / 20 |
| `severity` | 60.0% | 12 / 20 |

**Exact structured-row match:** 35.0% (7 / 20 rows) — all 8 structured fields correct (`risk_flags`/`supporting_image_ids` compared order-insensitively).

> Free-text fields `evidence_standard_met_reason` and `claim_status_justification` are not accuracy-scored (no single correct string); they are inspected qualitatively.

## claim_status confusion (rows: gold → predicted)

| gold ↓ / pred → | contradicted | not_enough_information | supported |
|---|---|---|---|
| **contradicted** | 2 | 3 | 0 |
| **not_enough_information** | 2 | 1 | 0 |
| **supported** | 0 | 0 | 12 |

`claim_status` accuracy: **75.0%**

## risk_flags (set-based, micro across rows; 'none' excluded)

- precision **62.9%**, recall **75.9%**, F1 **0.688** (TP=22, FP=13, FN=7)

## Operational analysis

- **Rows processed:** 20 (1 model call per claim)
- **Model calls (this run):** 20  •  cache hits: 0
- **Images sent:** 29
- **Tokens:** input 103,403 / output 10,058
- **Runtime:** 153.087 s
- **Sample-set cost:** $0.7685 (@ $5/1M in, $25/1M out)

### Projected test-set cost

`dataset/claims.csv` = **44 rows**, **82 images**. At 5170 in / 503 out tokens per claim (sample average):

- est. tokens: input 227,487 / output 22,128
- **est. cost: $1.6906** (full price)
- with the **Message Batches API (50% off)**: ~**$0.8453**; prompt-cache reads on the shared per-`claim_object` prefix cut input cost further; re-runs are ~$0 (on-disk cache).

### Throughput, rate limits & resilience

- **1 call/claim**, bounded concurrency; rows are grouped by `claim_object` so each group shares one cached prompt prefix (cache reads ≪ full input price).
- **Offline test run** uses the **Message Batches API (50% off)**; latency is not user-facing, so batch throughput matters more than per-call RPM/TPM.
- **Reproducibility:** Opus 4.8 rejects `temperature`; determinism comes from the on-disk response cache (keyed by prompt-version + model + claim + image bytes), so re-scoring a cached run is free and byte-identical. A *fresh* cold run can differ by a row or two (the model is not perfectly deterministic without temperature control); the cache pins the submitted `output.csv`.
- **Image handling:** media types are content-sniffed (dataset extensions lie); AVIF/HEIC are transcoded to PNG and any image over the API's 10 MiB cap is downscaled to a JPEG that fits.
- **Resilience:** bounded retries with backoff on transient API/parse errors; outputs are coerced onto closed enums and validated to the 14-column contract before write.
