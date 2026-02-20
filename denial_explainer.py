# ===== IMPORTS (tools we borrow) =====

import argparse      # lets us pass arguments from terminal (like --input file.json)
import json          # lets us read/write JSON data
import os            # lets us access environment variables (API key)
from anthropic import Anthropic   # official Claude API client

from core import validate_input, run_denial_explainer #let us access core.py
from pathlib import Path



# ===== SETTINGS / CONSTANTS =====
# These are values we may want to change later.

MODEL = "claude-sonnet-4-5"   # which Claude model to use
PROMPT_VERSION = "system_v2"

# System prompt = "boss instructions" for Claude
# These rules guide how Claude behaves. Moving system prompt out of Python strings
#PROMPT_VERSION = "system_v1"


def load_system_prompt(prompt_version: str) -> str:
    path = Path(__file__).parent / "prompts" / f"{prompt_version}.txt"
    return path.read_text(encoding="utf-8")

# Backward-compatible alias so older code can still import SYSTEM_PROMPT
SYSTEM_PROMPT = load_system_prompt(PROMPT_VERSION)


def load_fewshot() -> str:
    path = Path(__file__).parent / "prompts" / "fewshot_v1.txt"
    return path.read_text(encoding="utf-8")

def load_system_prompt(prompt_version: str) -> str:
    base = (Path(__file__).parent / "prompts" / f"{prompt_version}.txt").read_text(encoding="utf-8")
    return base + "\n\n" + load_fewshot()


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
- If carc_code or rarc_code is provided, incorporate the code meaning into your explanation.
- If carc_code or rarc_code is missing, list it in missing_information_needed.
"""



def build_user_prompt_loose(payload: dict) -> str:
    """Same as build_user_prompt but without word/array constraints. For temperature experiments only."""
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
- If carc_code or rarc_code is provided, incorporate the code meaning into your explanation.
- If carc_code or rarc_code is missing, list it in missing_information_needed.
"""

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

    # Validate input before calling the API
    errors = validate_input(denial_input)
    if errors:
        raise SystemExit(f"INPUT_INVALID: {errors}")


        
    client = Anthropic(api_key=api_key)

    #  Single reliable entry point (retry + parse + schema validation).
    data = run_denial_explainer(
        client,
        model=MODEL,
        system_prompt=load_system_prompt(PROMPT_VERSION),
        user_prompt=build_user_prompt(denial_input),
        max_tokens=900,
        temperature=0.0,
        prompt_version=PROMPT_VERSION,
    )


    # Print run metadata for traceability
    print(f"run_id: {data.get('_meta', {}).get('run_id')}")


    print(json.dumps(data, indent=2))



   
if __name__ == "__main__":
    main()

    
