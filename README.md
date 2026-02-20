# Denial Explainer (MVP)

Healthcare claim denials are common, expensive, and confusing—especially for members.  
This project is a safety-aware AI assistant that turns a denial reason (or codes) into:
- a plain-English explanation
- missing information needed to improve accuracy
- recommended next steps + an appeal checklist
- an optional appeal letter draft

The goal is to demonstrate **healthcare domain expertise + AI engineering + reliability practices**.

---

## Why it matters

Most denial explanations are hard to interpret and inconsistent across payers.  
A useful assistant must be:
- **clear for non-technical users**
- **conservative** (avoid hallucinating payer/CMS rules)
- **structured** (machine-readable output for workflows)
- **measurable** (evaluation + regression tests)

---

## Architecture (high level)


[User Input JSON]
    --> [Input Validation (core.py)]
    --> [CARC/RARC Code Lookup (code_lookup.py)]
    --> [Prompt Enrichment — inject official code descriptions]
    --> [Prompt Builder + Guardrails (denial_explainer.py)]
    --> [Claude API Call with retry/backoff (core.py)]
    --> [Extract JSON + Schema Validation (core.py)]
    --> [UI Output + Grounding Sources + Feedback (app.py)]


### Key design decisions

| Decision | Rationale |
|---|---|
| Structured JSON output (not free text) | Machine-readable for downstream workflows; enables automated evaluation |
| CARC/RARC grounding via local lookup | Deterministic, zero-latency, no hallucinated code meanings |
| Enrichment injected into prompt (not system prompt) | Keeps system prompt small; enrichment varies per request |
| Schema validation on every response | Catches truncation, missing keys, malformed output before showing to user |
| Retry with backoff | Handles transient API failures without user intervention |
| Temperature 0.0 in production | Deterministic output for healthcare use case; validated via temperature experiment |

---

## Project structure


denial-explainer/
├── app.py                     # Streamlit UI (single, compare, temperature experiment modes)
├── denial_explainer.py        # System prompt, user prompt builder, model config
├── core.py                    # Shared engine: validation, retry, JSON extraction, schema checks
├── code_lookup.py             # CARC/RARC lookup + prompt enrichment
├── eval_runner.py             # Evaluation framework (8 checks × N runs per case)
├── data/
│   ├── carc_codes.json        # 25 common CARC codes with descriptions + appeal guidance
│   └── rarc_codes.json        # 18 common RARC codes with action hints
├── eval_cases/
│   ├── standard/              # 7 real-world denial scenarios
│   └── adversarial/           # 5 adversarial cases (garbage, injection, hallucination, etc.)
├── eval_reports/              # Timestamped eval results (JSON + CSV)
├── feedback/                  # User feedback (feedback.jsonl)
├── prompts/                   # Versioned system prompts
└── requirements.txt


---

## Input format

The tool accepts claims denial data in a structured format that mirrors real ERA/EOB fields:

json
{
  "payer": "Anthem Blue Cross",
  "carc_code": "CO-50",
  "rarc_code": "N579",
  "denial_reason_text": "Non-covered service, not deemed medically necessary",
  "member_context": "Outpatient MRI of lumbar spine for chronic low back pain",
  "provider_context": "Ordering physician submitted clinical notes",
  "claim_type": "Outpatient",
  "codes": ["72148"],
  "diagnosis_codes": ["M54.5"],
  "place_of_service": "22",
  "dates": {"service_date": "2026-01-15"}
}


Only `carc_code` is required. All other fields improve accuracy and confidence.

---

## CARC/RARC Grounding (Phase 4)

Instead of relying on Claude's training data to interpret denial codes, the tool performs a **local lookup** against curated CARC/RARC tables before calling the API:

1. User enters `carc_code: "CO-50"` and optionally `rarc_code: "N579"`
2. `code_lookup.py` parses the code (handles formats: CO-50, CO50, co 50, CARC 50, 50)
3. Retrieves official X12/WPC description + appeal guidance
4. Injects enrichment into the prompt so Claude receives authoritative definitions
5. Claude cites sources in `sources_used` output field

**Why this matters:** A generic LLM might hallucinate code meanings or invent policy details. Grounding with authoritative data ensures the explanation matches the official code definition. For codes not in the lookup table, the system flags them as unrecognized rather than guessing.

Coverage: 25 CARC codes + 18 RARC codes (most common denial-related codes). The lookup table is extensible — adding codes requires only a JSON edit, no code changes.

---

## Evaluation Framework (Phase 3)

The eval framework runs every test case through Claude and validates the output against domain-informed expectations.

### 8 checks per run:
| Check | What it validates |
|---|---|
| schema_valid | All required keys present, correct types |
| confidence_band | Confidence within expected range (e.g., low for sparse input) |
| must_include | Required keywords appear (e.g., "authorization" for auth denials) |
| must_include_any | At least one synonym appears (flexible keyword matching) |
| must_not_include | Forbidden content absent (blocks hallucinated CFR/CMS citations) |
| missing_info | Missing information count meets expectations |
| next_steps_count | Minimum actionable steps provided |
| root_causes_count | Minimum root causes identified |

### Consistency testing:
Each case runs 3 times to verify deterministic output at temperature 0.0. Checks confidence consistency and root cause overlap (Jaccard similarity).

### Test case coverage:

**Standard cases (7):** Medical necessity (CO-50), prior authorization (CO-15), eligibility (CO-27), timely filing (CO-29), coordination of benefits (OA-23), coding/modifier (CO-4), Medicare/LCD (CO-50 + N386)

**Adversarial cases (5):** Garbage input, contradictory fields, prompt injection (fake CFR citations), hallucination trap (fake CARC code), minimal/empty input

### Running evals:

python eval_runner.py --dry-run          # validate case files
python eval_runner.py --runs 1           # quick single run (~$0.14)
python eval_runner.py --runs 3           # full consistency check (~$0.42)
python eval_runner.py --case 001         # single case
python eval_runner.py --category adversarial  # adversarial only


### Latest results:

Prompt version : system_v2
Model          : claude-sonnet-4-5
12/12 passed, 0 failures
Cost per explanation: ~$0.02


---

## Prompt engineering

### System prompt versioning:
- `system_v1` — Baseline with guardrails (no CARC/RARC awareness)
- `system_v2` — Added CARC/RARC field awareness, tightened hallucination guardrails (no echoing regulatory citations from input)

### Key guardrails:
- Never cite CFR/CMS sections — even if they appear in user input (prevents prompt injection)
- Never invent meanings for unrecognized CARC/RARC codes
- Reduce confidence when input is incomplete
- Flag contradictory input fields
- Medicare/Medicaid: always note MAC/state variability

### Temperature experiment findings:
Tested temperatures 0.0, 0.3, and 0.7 with both constrained and unconstrained prompts. Minimal output variation observed — expected for structured JSON extraction with tight schema constraints. **Recommendation:** temperature 0.0 for deterministic, reproducible results.

---

## Reliability layer (Phase 1)

Built into `core.py`:
- **Input validation** — required fields, type checking, claim_type enum
- **Retry with exponential backoff** — handles transient API failures (2 retries)
- **JSON extraction** — parses Claude's response even with minor formatting issues
- **Schema validation** — checks all required keys, correct types, word/item limits
- **Error taxonomy** — categorized errors (INPUT_INVALID, MODEL_TRUNCATED, MODEL_NO_JSON, API_ERROR) with run_id for debugging
- **Cost tracking** — input/output tokens and estimated cost per call

---

## Safety / PHI stance

- Do not paste PHI (member name, DOB, member ID, address).
- Designed to be conservative: if inputs are incomplete, it asks for missing details.
- The assistant must not invent payer policy specifics or cite CMS/CFR references unless provided.
- Output is educational/workflow support and not medical or legal advice.

---

## Cost

| Operation | Estimated cost |
|---|---|
| Single explanation | ~$0.02 |
| Full eval (12 cases × 3 runs) | ~$0.72 |
| Model | Claude Sonnet 4.5 ($3/MTok input, $15/MTok output) |

Prompt caching enabled on system prompt for repeated calls within 5-minute window.

---

## How to run (Windows)

### Setup

# Create and activate venv
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API key
set ANTHROPIC_API_KEY=your-key-here


### Run the Streamlit UI

streamlit run app.py


### Run evaluations

python eval_runner.py --dry-run    # validate test cases
python eval_runner.py --runs 1     # quick eval run
python eval_runner.py --runs 3     # full consistency check


### Run code lookup standalone

python code_lookup.py              # tests CARC/RARC parsing and lookup


---

## Roadmap

- [x] **Phase 0** — Baseline MVP, Git setup, README
- [x] **Phase 1** — Reliability layer (retry, validation, error taxonomy)
- [x] **Phase 2** — Prompt engineering (versioning, guardrails, temperature experiment)
- [x] **Phase 3** — Evaluation framework (12 cases, 8 checks, consistency testing)
- [x] **Phase 4** — CARC/RARC grounding (local lookup, prompt enrichment, sources_used)
- [ ] **Phase 5** — Full RAG (PDF ingestion, embeddings, vector store, citations)
- [ ] **Phase 6** — Expanded feedback loop (categorized issues, prompt improvement cycle)