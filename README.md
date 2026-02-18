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

[User Input JSON] --> [Validation] --> [Prompt Builder + Guardrails]
--> [Claude API Call (retry/backoff)]
--> [Extract JSON] --> [Schema Validate]
--> [UI Output + Download + History + Feedback (later)]



> FUTURE REFERENCE: Keeping logic modular lets CLI + UI share the same “core” reliability layer.

---

## Safety / PHI stance

- Do not paste PHI (member name, DOB, member ID, address).
- Designed to be conservative: if inputs are incomplete, it asks for missing details.
- The assistant must not invent payer policy specifics or cite CMS/CFR references unless provided.
- Output is educational/workflow support and not medical or legal advice.
---

## How to run (Windows)

### Setup
```bash
# Create and activate venv
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt





