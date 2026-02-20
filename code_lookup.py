"""
code_lookup.py — CARC/RARC Code Lookup and Prompt Enrichment

Loads local CARC and RARC code tables and provides:
1. Code description lookup
2. Prompt enrichment — injects official descriptions into the user prompt
   so Claude doesn't have to guess or hallucinate code meanings.

Usage:
    from code_lookup import enrich_input, lookup_carc, lookup_rarc

    # Enrich a denial input dict before sending to Claude
    enriched = enrich_input(denial_input)

    # Or look up individual codes
    info = lookup_carc("50")      # returns dict with description, guidance, etc.
    info = lookup_rarc("N579")    # returns dict with description, category
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Load data files
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"


def _load_json(filename):
    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"WARNING: {filepath} not found. Code lookup will be unavailable.")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


_CARC_DATA = _load_json("carc_codes.json")
_RARC_DATA = _load_json("rarc_codes.json")

CARC_CODES = _CARC_DATA.get("carc_codes", {})
RARC_CODES = _RARC_DATA.get("rarc_codes", {})
GROUP_CODES = _CARC_DATA.get("group_codes", {})


# ---------------------------------------------------------------------------
# Parse CARC code from various formats
# ---------------------------------------------------------------------------
def parse_carc_code(raw: str) -> tuple:
    """
    Parse a CARC code string into (group, number).
    Handles: 'CO-50', 'CO50', 'CO 50', 'CARC 50', '50', 'co-50'

    Returns: (group_code or None, number_str)
    """
    if not raw or not isinstance(raw, str):
        return None, None

    raw = raw.strip().upper()

    # Pattern: GROUP-NUMBER or GROUP NUMBER or GROUPNUMBER
    match = re.match(r"^(CO|PR|OA|PI|CR)[- ]?(\d+)$", raw)
    if match:
        return match.group(1), match.group(2)

    # Pattern: CARC NUMBER or just NUMBER
    match = re.match(r"^(?:CARC[- ]?)?(\d+)$", raw)
    if match:
        return None, match.group(1)

    return None, None


def parse_rarc_code(raw: str) -> str:
    """
    Normalize a RARC code string.
    Handles: 'N579', 'n579', 'RARC N579', 'MA01'

    Returns: normalized code string or None
    """
    if not raw or not isinstance(raw, str):
        return None

    raw = raw.strip().upper()

    # Remove RARC prefix if present
    match = re.match(r"^(?:RARC[- ]?)?([A-Z]+\d+)$", raw)
    if match:
        return match.group(1)

    return None


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------
def lookup_carc(raw_code: str) -> dict:
    """
    Look up a CARC code and return its details.
    Returns dict with: found, code, group, group_description, description,
                       category, appeal_guidance
    """
    group, number = parse_carc_code(raw_code)

    if number is None:
        return {"found": False, "raw_input": raw_code, "error": "Could not parse CARC code"}

    info = CARC_CODES.get(number)

    if not info:
        return {
            "found": False,
            "raw_input": raw_code,
            "group": group,
            "number": number,
            "error": f"CARC {number} not in local lookup table"
        }

    result = {
        "found": True,
        "raw_input": raw_code,
        "group": group,
        "number": number,
        "description": info.get("description", ""),
        "category": info.get("category", ""),
        "appeal_guidance": info.get("appeal_guidance", ""),
        "common_group": info.get("common_group", ""),
    }

    # Add group description
    effective_group = group or info.get("common_group")
    if effective_group and effective_group in GROUP_CODES:
        result["group_description"] = GROUP_CODES[effective_group]

    return result


def lookup_rarc(raw_code: str) -> dict:
    """
    Look up a RARC code and return its details.
    Returns dict with: found, code, description, category
    """
    normalized = parse_rarc_code(raw_code)

    if not normalized:
        return {"found": False, "raw_input": raw_code, "error": "Could not parse RARC code"}

    info = RARC_CODES.get(normalized)

    if not info:
        return {
            "found": False,
            "raw_input": raw_code,
            "code": normalized,
            "error": f"RARC {normalized} not in local lookup table"
        }

    return {
        "found": True,
        "raw_input": raw_code,
        "code": normalized,
        "description": info.get("description", ""),
        "category": info.get("category", ""),
    }


# ---------------------------------------------------------------------------
# Enrich input payload
# ---------------------------------------------------------------------------
def enrich_input(payload: dict) -> dict:
    """
    Take a denial input dict and add code lookup results.
    Adds _enrichment dict with CARC/RARC descriptions and guidance.
    Does NOT modify the original payload — returns a new dict.

    This enriched dict gets passed to build_user_prompt() so Claude
    receives the official code meanings instead of having to guess.
    """
    enriched = dict(payload)  # shallow copy
    enrichment = {}

    # CARC lookup
    carc_raw = payload.get("carc_code", "")
    if carc_raw:
        carc_info = lookup_carc(carc_raw)
        enrichment["carc_lookup"] = carc_info

        if carc_info.get("found"):
            enrichment["carc_official_description"] = carc_info["description"]
            enrichment["carc_category"] = carc_info["category"]
            enrichment["carc_appeal_guidance"] = carc_info["appeal_guidance"]
            if carc_info.get("group_description"):
                enrichment["carc_group_meaning"] = carc_info["group_description"]

    # RARC lookup
    rarc_raw = payload.get("rarc_code", "")
    if rarc_raw:
        rarc_info = lookup_rarc(rarc_raw)
        enrichment["rarc_lookup"] = rarc_info

        if rarc_info.get("found"):
            enrichment["rarc_official_description"] = rarc_info["description"]
            enrichment["rarc_category"] = rarc_info["category"]

    # Add enrichment to payload
    if enrichment:
        enriched["_code_enrichment"] = enrichment

    return enriched


# ---------------------------------------------------------------------------
# Generate sources_used list
# ---------------------------------------------------------------------------
def get_sources_used(payload: dict) -> list:
    """
    Return a list of sources used for grounding, based on what was looked up.
    Intended for inclusion in the output JSON.
    """
    sources = []
    enrichment = payload.get("_code_enrichment", {})

    carc_info = enrichment.get("carc_lookup", {})
    if carc_info.get("found"):
        sources.append(f"CARC {carc_info['number']}: {carc_info['description'][:80]} (X12/WPC)")

    rarc_info = enrichment.get("rarc_lookup", {})
    if rarc_info.get("found"):
        sources.append(f"RARC {rarc_info['code']}: {rarc_info['description'][:80]} (X12/WPC)")

    return sources


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== CARC Lookup Test ===")
    for code in ["CO-50", "co 50", "CARC 50", "50", "PR-1", "CO-15", "CO-9999", "garbage"]:
        result = lookup_carc(code)
        status = "FOUND" if result.get("found") else "NOT FOUND"
        print(f"  {code:12s} -> {status}: {result.get('description', result.get('error', ''))[:60]}")

    print("\n=== RARC Lookup Test ===")
    for code in ["N579", "MA01", "n579", "RARC N386", "XYZ99", ""]:
        result = lookup_rarc(code)
        status = "FOUND" if result.get("found") else "NOT FOUND"
        print(f"  {code:12s} -> {status}: {result.get('description', result.get('error', ''))[:60]}")

    print("\n=== Enrich Test ===")
    test_input = {
        "carc_code": "CO-50",
        "rarc_code": "N579",
        "denial_reason_text": "Not medically necessary",
    }
    enriched = enrich_input(test_input)
    print(json.dumps(enriched.get("_code_enrichment", {}), indent=2))

    print("\n=== Sources Used ===")
    sources = get_sources_used(enriched)
    for s in sources:
        print(f"  - {s}")
