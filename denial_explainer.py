# ===== IMPORTS (tools we borrow) =====

import argparse      # lets us pass arguments from terminal (like --input file.json)
import json          # lets us read/write JSON data
import os            # lets us access environment variables (API key)
from anthropic import Anthropic   # official Claude API client


# ===== SETTINGS / CONSTANTS =====
# These are values we may want to change later.

MODEL = "claude-sonnet-4-5"   # which Claude model to use

# System prompt = "boss instructions" for Claude
# These rules guide how Claude behaves.
SYSTEM_PROMPT = """You are a healthcare claims denial explainer for US health insurance.
You must be accurate, conservative, and safe.

Rules:
- If information is missing, say so explicitly and ask for the minimum extra fields needed.
- Do NOT invent policy details or CPT/ICD rules.
- You must output ONLY a JSON object. Do not wrap in ``` fences. Do not add any extra text.
- The first character of your entire response must be { and the last character must be }.
- Keep language plain and member-friendly.
"""


# ===== BUILD THE USER PROMPT =====
# This function builds the actual question Claude receives.

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
- Do NOT use words like "5th grade" or "8th grade" in output.
- If you are unsure, set confidence=low and list missing_information_needed.
- Do not mention internal model details.
- Keep each array to at most 4 items.
- Each item must be <= 18 words.
- plain_english_explanation must be <= 60 words.
- recommended_next_steps must be 4 items max.
- appeal_checklist must be 4 items max.
- risk_warnings must be 3 items max.
"""


# ===== EXTRACT JSON SAFELY =====
# Claude sometimes adds extra text.
# This function pulls only the JSON part.

def extract_json(text: str) -> str:
    if not text:
        return ""

    t = text.strip()

    # Remove ``` code fences if Claude added them
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        if t.endswith("```"):
            t = t[:-3].strip()

    # Extract text between first { and last }
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""

    return t[start : end + 1]


# ===== LOAD INPUT FILE =====
# Reads JSON file and converts to Python dictionary.

def load_input(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===== MAIN PROGRAM =====

def main():

    # Get API key from environment variable (secure)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set.")


    # Allow command-line input
    parser = argparse.ArgumentParser(description="Claude-powered healthcare denial explainer (MVP).")
    parser.add_argument("--input", type=str, default="", help="Path to a JSON file (e.g., samples/denial_1.json)")
    args = parser.parse_args()

    # Decide where denial input comes from

    if args.input:
        denial_input = load_input(args.input)
    else:
        denial_input = {
            "payer": "Example Health Plan",
            "denial_code_or_reason": "Service not medically necessary",
            "member_context": "Outpatient imaging for back pain",
            "provider_context": "Ordering physician submitted clinical notes",
            "dates": {"service_date": "2026-02-01"},
        }

    # Create Claude client
    client = Anthropic(api_key=api_key)

    # Send prompt to Claude
    resp = client.messages.create(
        model=MODEL,
        max_tokens=900,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(denial_input)}],
    )

    # Robust extraction across SDK versions: take any block that has .text
    # Extract text from Claude response blocks
    text_parts = []
    for block in resp.content:
        txt = getattr(block, "text", None)
        if isinstance(txt, str) and txt.strip():
            text_parts.append(txt)

    text = "\n".join(text_parts).strip()

    # Fail fast if output is truncated
    if "}" not in text:
        raise SystemExit(
            "Output appears truncated (missing closing brace). "
            "Increase max_tokens or shorten constraints."
        )

    # Extract JSON portion
    json_text = extract_json(text)
    if not json_text:
        print("\n--- RAW MODEL OUTPUT START ---\n")
        print(repr(text))
        print("\n--- RAW MODEL OUTPUT END ---\n")
        raise SystemExit("Could not find a JSON object in model output.")

    # Convert JSON string → Python dictionary
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        print("\n--- RAW MODEL OUTPUT START ---\n")
        print(repr(text))
        print("\n--- RAW MODEL OUTPUT END ---\n")
        print("\n--- EXTRACTED JSON START ---\n")
        print(repr(json_text))
        print("\n--- EXTRACTED JSON END ---\n")
        raise SystemExit(f"JSON parse error: {e}")

    # Print clean formatted output
    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    main()
