import json
import os
from anthropic import Anthropic
from anthropic.types import TextBlock

MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are a healthcare claims denial explainer for US health insurance.
You must be accurate, conservative, and safe.

Rules:
- If information is missing, say so explicitly and ask for the minimum extra fields needed.
- Do NOT invent policy details or CPT/ICD rules.
- You must output ONLY a JSON object. Do not wrap in ``` fences. Do not add any extra text.
- The first character of your entire response must be { and the last character must be }.
- Keep language plain and member-friendly.
"""

def build_user_prompt(payload: dict) -> str:
    return f"""Explain this claim denial in plain language and suggest next steps.

Input JSON:
{json.dumps(payload, indent=2)}

Return JSON with exactly this schema:
{{
  "plain_english_explanation": "string",
  "likely_root_causes": ["string", "..."],
  "missing_information_needed": ["string", "..."],
  "recommended_next_steps": ["string", "..."],
  "appeal_checklist": ["string", "..."],
  "risk_warnings": ["string", "..."],
  "confidence": "low|medium|high"
}}

Constraints:
- Use 5th–8th grade language in plain_english_explanation.
- Do NOT use words like 5th–8th grade in output.
- If you are unsure, set confidence=low and list missing_information_needed.
- Do not mention internal model details.
- Keep each array to at most 4 items.
- Each item must be <= 18 words.
- plain_english_explanation must be <= 60 words.
- recommended_next_steps must be 4 items max.
- appeal_checklist must be 4 items max.
- risk_warnings must be 3 items max.
"""

def extract_json(text: str) -> str:
    """
    Best-effort extraction of a JSON object from model output.
    Handles common cases like:
    - leading/trailing prose
    - ```json fenced blocks
    - leading BOM/whitespace
    """
    if not text:
        return ""

    t = text.strip()

    # Remove fenced code blocks if present
    if t.startswith("```"):
        # Strip first fence line (``` or ```json)
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        # Strip closing fence
        if t.endswith("```"):
            t = t[:-3].strip()

    # Find first '{' and last '}' to extract object
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""

    return t[start : end + 1]



def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set.")

    # Minimal sample input (edit this as you like)
    denial_input = {
        "payer": "Example Health Plan",
        "denial_code_or_reason": "Service not medically necessary",
        "member_context": "Outpatient imaging for back pain",
        "provider_context": "Ordering physician submitted clinical notes",
        "dates": {"service_date": "2026-02-01"},
    }

    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(denial_input)}],
    )

    # Join all text blocks safely (sometimes content has multiple blocks)
    text_parts = []
    for block in resp.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)

    text = "\n".join(text_parts).strip()
    # Debug output (keep for now)
    print("\n--- RAW MODEL OUTPUT START ---\n")
    print(repr(text))
    print("\n--- RAW MODEL OUTPUT END ---\n")

    # 🔴 Fail fast if Claude output is truncated
    if "}" not in text:
        raise SystemExit(
            "Output appears truncated (missing closing brace). "
            "Increase max_tokens or shorten constraints."
        )


    json_text = extract_json(text)
    if not json_text:
        print("\n--- RAW MODEL OUTPUT START ---\n")
        print(repr(text))
        print("\n--- RAW MODEL OUTPUT END ---\n")
        raise SystemExit("Could not find a JSON object in model output.")

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
   
        DEBUG = False
        if DEBUG:
            print("\n--- RAW MODEL OUTPUT START ---\n")
            print(repr(text))
            print("\n--- RAW MODEL OUTPUT END ---\n")
        
            print("\n--- EXTRACTED JSON START ---\n")
            print(repr(json_text))
            print("\n--- EXTRACTED JSON END ---\n")
            print(f"JSON parse error: {e}")
            raise



    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
