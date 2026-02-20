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
# Expanded for CARC/RARC and plan types.
REQUIRED_INPUT_KEYS = {"carc_code"}

OPTIONAL_INPUT_KEYS = {
    "rarc_code": str,
    "denial_reason_text": str,
    "payer": str,
    "member_context": str,
    "provider_context": str,
    "claim_type": str,
    "codes": list,
    "diagnosis_codes": list,
    "place_of_service": str,
    "dates": dict,
}

VALID_CLAIM_TYPES = {"professional", "outpatient", "inpatient", "dme"}

def validate_input(payload: dict) -> list:
    errors = []
    if not isinstance(payload, dict):
        return ["Input must be a JSON object"]

    # Accept old or new format (backward compat during migration)
    has_new = "carc_code" in payload
    has_old = "denial_code_or_reason" in payload

    if not has_new and not has_old:
        errors.append("Missing required field: carc_code (or legacy denial_code_or_reason)")

    # Type check optional fields if present
    for key, expected_type in OPTIONAL_INPUT_KEYS.items():
        if key in payload and not isinstance(payload[key], expected_type):
            errors.append(f"{key} should be {expected_type.__name__}, got {type(payload[key]).__name__}")

    # Validate claim_type if present
    if "claim_type" in payload:
        if payload["claim_type"].lower() not in VALID_CLAIM_TYPES:
            errors.append(f"claim_type should be one of {VALID_CLAIM_TYPES}")

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
    "sources_used",
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

    # Lists expected + item caps from prompt
    list_caps = {
        "likely_root_causes": 4,
        "missing_information_needed": 4,
        "recommended_next_steps": 4,
        "appeal_checklist": 4,
        "risk_warnings": 3,
        "sources_used": 5,
    }

    for key, cap in list_caps.items():
        val = data.get(key)
        if not isinstance(val, list):
            errors.append(f"{key} should be a list")
            continue
        if len(val) > cap:
            errors.append(f"{key} has {len(val)} items (max {cap})")

        # Each item <= 18 words
        for i, item in enumerate(val):
            if not isinstance(item, str):
                errors.append(f"{key}[{i}] must be a string")
                continue
            if len(item.split()) > 18:
                errors.append(f"{key}[{i}] exceeds 18 words")

    # Plain-English explanation expected + <= 60 words
    pe = data.get("plain_english_explanation")
    if not isinstance(pe, str):
        errors.append("plain_english_explanation should be a string")
    else:
        if len(pe.split()) > 60:
            errors.append("plain_english_explanation exceeds 60 words")

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


def extract_and_parse_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON object from model text and parse it.
    Raises RuntimeError with taxonomy codes for clean handling in UI/CLI.
    """
    json_text = extract_json(text)
    if not json_text:
        raise RuntimeError(f"{ERR_MODEL_NO_JSON}: could not find JSON object")

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"{ERR_MODEL_JSON_PARSE}: {e}")


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
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
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
    prompt_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Single entry point used by UI + CLI.

    Steps:
    1) create run_id
    2) call Claude (retry)
    3) check truncation
    4) extract + parse JSON
    5) validate schema
    6) attach safe metadata + log completion
    """
    run_id = str(uuid.uuid4())
    logger.info("Starting run run_id=%s model=%s max_tokens=%s temperature=%s prompt_version=%s",
                run_id, model, max_tokens, temperature, prompt_version)

    raw_text, resp = call_claude_with_retry(
        client,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if "}" not in raw_text:
        raise RuntimeError(f"{ERR_MODEL_TRUNCATED}: run_id={run_id}: missing closing brace")

    data = extract_and_parse_json(raw_text)

    errors = validate_response(data)
    if errors:
        raise RuntimeError(f"{ERR_MODEL_SCHEMA_INVALID}: run_id={run_id}: {errors}")

    usage_obj = getattr(resp, "usage", None)
    usage = None
    if usage_obj is not None:
        usage = {
            "input_tokens": getattr(usage_obj, "input_tokens", None),
            "output_tokens": getattr(usage_obj, "output_tokens", None),
            "cache_read_input_tokens": getattr(usage_obj, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(usage_obj, "cache_creation_input_tokens", None),
        }

    data["_meta"] = {
        "run_id": run_id,
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "prompt_version": prompt_version,
        "usage": usage,
    }

    logger.info(
        "Completed run run_id=%s model=%s input_tokens=%s output_tokens=%s confidence=%s",
        run_id,
        model,
        (usage or {}).get("input_tokens"),
        (usage or {}).get("output_tokens"),
        data.get("confidence"),
    )

    return data


    meta = data.get("_meta") or {}
    usage = meta.get("usage") or {}

    logger.info(
        "Completed run run_id=%s model=%s input_tokens=%s output_tokens=%s confidence=%s",
        meta.get("run_id", run_id),
        meta.get("model", model),
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        data.get("confidence"),
    )


