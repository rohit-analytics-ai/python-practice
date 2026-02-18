"""
core.py is the shared "engine" used by both:
- denial_explainer.py (CLI)
- app.py (Streamlit UI)

Goal:
- one place for validation, retries, parsing, and schema checks
- consistent behavior across CLI + UI
"""

import json
import time
import uuid
from typing import Any, Dict, List, Tuple, Optional
import logging
from pathlib import Path

# ===== Error taxonomy (clean UI + logs) =====
# Standard error codes make debugging + eval reporting much easier.
ERR_INPUT_INVALID = "INPUT_INVALID"
ERR_API_RETRY_EXHAUSTED = "API_RETRY_EXHAUSTED"
ERR_MODEL_TRUNCATED = "MODEL_TRUNCATED"
ERR_MODEL_NO_JSON = "MODEL_NO_JSON"
ERR_MODEL_SCHEMA_INVALID = "MODEL_SCHEMA_INVALID"
ERR_MODEL_JSON_PARSE = "MODEL_JSON_PARSE"


# Basic file logging (avoid logging raw PHI; log only metadata)
LOG_PATH = Path("denial_explainer.log")
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("denial_explainer")


# ===== Input validation =====
# Start strict on minimum viable fields; expand later as we add CARC/RARC and plan types.
REQUIRED_INPUT_KEYS = {"denial_code_or_reason"}  # minimum viable


def validate_input(payload: Any) -> List[str]:
    """
    Validate user input payload before sending to the model.

    Returns: list of error strings (empty list means valid)
    """
    errors: List[str] = []

    if not isinstance(payload, dict):
        errors.append("Input must be a JSON object (dictionary).")
        return errors

    missing = REQUIRED_INPUT_KEYS - set(payload.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")

    return errors


# ===== Output validation =====
REQUIRED_OUTPUT_KEYS = {
    "plain_english_explanation",
    "likely_root_causes",
    "missing_information_needed",
    "recommended_next_steps",
    "appeal_checklist",
    "risk_warnings",
    "confidence",
}


def validate_response(data: Any) -> List[str]:
    """
    Validate the model response schema to avoid trusting hallucinated / malformed outputs.

    Returns: list of error strings (empty list means valid)
    """
    errors: List[str] = []

    if not isinstance(data, dict):
        errors.append("Model output must be a JSON object.")
        return errors

    missing = REQUIRED_OUTPUT_KEYS - set(data.keys())
    if missing:
        errors.append(f"Missing keys: {sorted(missing)}")

    confidence = data.get("confidence")
    if confidence not in ("low", "medium", "high"):
        errors.append(f"Invalid confidence: {confidence!r} (must be low|medium|high)")

    # Lists expected
    for key in [
        "likely_root_causes",
        "missing_information_needed",
        "recommended_next_steps",
        "appeal_checklist",
        "risk_warnings",
    ]:
        val = data.get(key)
        if not isinstance(val, list):
            errors.append(f"{key} should be a list")

    # Plain-English explanation expected
    if not isinstance(data.get("plain_english_explanation"), str):
        errors.append("plain_english_explanation should be a string")

    return errors


# ===== JSON extraction =====
def extract_json(text: str) -> str:
    """
    FUTURE REFERENCE:
    Even when we instruct JSON-only, models may add prose or code fences.
    This function extracts the first JSON object between '{' and '}'.
    """
    if not text:
        return ""

    t = text.strip()

    # Remove fenced code blocks if present
    if t.startswith("```"):
        first_newline = t.find("\n")
        if first_newline != -1:
            t = t[first_newline + 1 :]
        if t.endswith("```"):
            t = t[:-3].strip()

    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""

    return t[start : end + 1]


# ===== Claude call with retry =====
def call_claude_with_retry(
    client,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: Optional[float] = None,
    max_retries: int = 2,
) -> Tuple[str, Any]:
    """
    Calls Claude with basic retry/backoff.

    Returns:
      - raw_text (combined text blocks)
      - resp (original SDK response object for usage metadata)
    """
    last_err: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            kwargs = dict(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # Some SDKs support temperature; keep optional to avoid breakage.
            if temperature is not None:
                kwargs["temperature"] = temperature

            resp = client.messages.create(**kwargs)

            # Robust extraction: join any content blocks that have a .text attribute
            text_parts: List[str] = []
            for block in resp.content:
                txt = getattr(block, "text", None)
                if isinstance(txt, str) and txt.strip():
                    text_parts.append(txt)

            raw_text = "\n".join(text_parts).strip()
            return raw_text, resp

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s...
                continue
            break

    raise RuntimeError(f"{ERR_API_RETRY_EXHAUSTED}: {last_err}")


# ===== End-to-end runner =====
def run_denial_explainer(
    client,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: Optional[float] = None,
) -> Dict[str, Any]:
    """
    FUTURE REFERENCE:
    This is the single entry point both UI + CLI will use.

    Steps:
    1) call Claude (retry)
    2) check truncation
    3) extract JSON
    4) parse JSON
    5) validate schema
    """

    logger.info("Starting run model=%s max_tokens=%s temperature=%s", model, max_tokens, temperature)


    raw_text, resp = call_claude_with_retry(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if "}" not in raw_text:
        raise RuntimeError(f"{ERR_MODEL_TRUNCATED}: missing closing brace")

    json_text = extract_json(raw_text)
    if not json_text:
        raise RuntimeError(f"{ERR_MODEL_NO_JSON}: could not find JSON object")

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{ERR_MODEL_JSON_PARSE}: {e}")

    errors = validate_response(data)
    if errors:
        raise RuntimeError(f"{ERR_MODEL_SCHEMA_INVALID}: {errors}")

    # Attach metadata for observability/cost tracking (safe to show)
    usage_obj = getattr(resp, "usage", None)

    # Convert SDK Usage object to a plain dict (safe for JSON)
    # In production, dont store SDK objects directly inside JSON,  
    # instead normalize them into plain primitives (str/int/bool/list/dict).

    usage = None
    if usage_obj is not None:
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", None),
            "output_tokens": getattr(usage_obj, "output_tokens", None),
            "cache_read_input_tokens": getattr(usage_obj, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(usage_obj, "cache_creation_input_tokens", None),
        }

    data["_meta"] = {
        "run_id": str(uuid.uuid4()),
        "model": model,
        "max_tokens": max_tokens,
        "usage": usage,
    }


    return data

    meta = data.get("_meta", {})
    logger.info(
        "Completed run run_id=%s model=%s input_tokens=%s output_tokens=%s confidence=%s",
        meta.get("run_id"),
        meta.get("model"),
        (meta.get("usage") or {}).get("input_tokens"),
        (meta.get("usage") or {}).get("output_tokens"),
        data.get("confidence"),
    )

