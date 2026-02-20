"""
eval_runner.py — Denial Explainer Evaluation Framework

Runs all eval cases (standard + adversarial), checks against expectations,
measures consistency across repeated runs, and produces reports.

Usage:
    python eval_runner.py                          # run all cases, 3 runs each
    python eval_runner.py --runs 1                 # quick single run (faster, cheaper)
    python eval_runner.py --category adversarial   # only adversarial cases
    python eval_runner.py --case 001               # run a single test case by ID
    python eval_runner.py --dry-run                # validate case files without calling API
"""

import argparse
import csv
import datetime
import glob
import json
import os
import sys
import time
from pathlib import Path

from anthropic import Anthropic
from denial_explainer import build_user_prompt, SYSTEM_PROMPT, MODEL, PROMPT_VERSION
from core import validate_input, run_denial_explainer
from code_lookup import enrich_input

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EVAL_CASES_DIR = Path(__file__).resolve().parent / "eval_cases"
EVAL_OUTPUT_DIR = Path(__file__).resolve().parent / "eval_reports"
MAX_TOKENS = 900
TEMPERATURE = 0.0
DEFAULT_RUNS = 3  # how many times to run each case for consistency check


# ---------------------------------------------------------------------------
# Load test cases
# ---------------------------------------------------------------------------
def load_all_cases(category_filter=None, case_filter=None):
    """Load all JSON test case files from eval_cases/ subfolders."""
    cases = []
    patterns = [
        str(EVAL_CASES_DIR / "standard" / "*.json"),
        str(EVAL_CASES_DIR / "adversarial" / "*.json"),
    ]

    for pattern in patterns:
        for filepath in sorted(glob.glob(pattern)):
            with open(filepath, "r", encoding="utf-8") as f:
                try:
                    case = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"  WARNING: Skipping {filepath} (invalid JSON: {e})")
                    continue

            case["_filepath"] = filepath

            # Apply filters
            if case_filter and case.get("test_id") != case_filter:
                continue
            if category_filter:
                cat = case.get("category", "")
                if category_filter == "adversarial" and not cat.startswith("adversarial"):
                    continue
                if category_filter == "standard" and cat.startswith("adversarial"):
                    continue
                if category_filter not in ("standard", "adversarial") and category_filter != cat:
                    continue

            cases.append(case)

    return cases


# ---------------------------------------------------------------------------
# Validation: check case file structure
# ---------------------------------------------------------------------------
def validate_case_file(case):
    """Check that a test case has required fields. Returns list of errors."""
    errors = []
    if "test_id" not in case:
        errors.append("missing test_id")
    if "input" not in case:
        errors.append("missing input")
    if "expectations" not in case:
        errors.append("missing expectations")
    elif not isinstance(case["expectations"], dict):
        errors.append("expectations must be a dict")
    return errors


# ---------------------------------------------------------------------------
# Evaluation checks
# ---------------------------------------------------------------------------
REQUIRED_RESPONSE_KEYS = {
    "plain_english_explanation",
    "likely_root_causes",
    "missing_information_needed",
    "recommended_next_steps",
    "appeal_checklist",
    "risk_warnings",
    "confidence",
    "sources_used",
}


def check_schema_valid(data):
    """Check that response has all required keys and correct types."""
    if not isinstance(data, dict):
        return False, "response is not a dict"

    data_keys = {k for k in data.keys() if k != "_meta"}
    missing = REQUIRED_RESPONSE_KEYS - data_keys
    if missing:
        return False, f"missing keys: {missing}"

    if data.get("confidence") not in ("low", "medium", "high"):
        return False, f"invalid confidence: {data.get('confidence')}"

    for key in ["likely_root_causes", "recommended_next_steps",
                "appeal_checklist", "risk_warnings", "missing_information_needed","sources_used"]:
        if not isinstance(data.get(key, []), list):
            return False, f"{key} is not a list"

    if not isinstance(data.get("plain_english_explanation", ""), str):
        return False, "plain_english_explanation is not a string"

    return True, "ok"


def check_confidence_band(data, expectations):
    """Check if confidence is within expected band."""
    expected = expectations.get("confidence_band")
    if not expected:
        return True, "no confidence expectation set"

    actual = data.get("confidence", "").lower()
    if actual in expected:
        return True, f"{actual} in {expected}"
    return False, f"expected {expected}, got {actual}"


def check_must_include(data, expectations):
    """Check that required keywords appear somewhere in the response."""
    keywords = expectations.get("must_include_keywords", [])
    if not keywords:
        return True, "no must_include keywords set"

    # Flatten all text content from the response
    text_parts = []
    text_parts.append(data.get("plain_english_explanation", ""))
    for key in ["likely_root_causes", "recommended_next_steps",
                "appeal_checklist", "risk_warnings", "missing_information_needed"]:
        text_parts.extend(data.get(key, []))

    full_text = " ".join(str(p) for p in text_parts).lower()

    missing = []
    for kw in keywords:
        if kw.lower() not in full_text:
            missing.append(kw)

    if missing:
        return False, f"missing keywords: {missing}"
    return True, "all keywords found"


def check_must_include_any(data, expectations):
    """Check that AT LEAST ONE of the keywords appears in the response."""
    keywords = expectations.get("must_include_any_keywords", [])
    if not keywords:
        return True, "no must_include_any keywords set"

    text_parts = []
    text_parts.append(data.get("plain_english_explanation", ""))
    for key in ["likely_root_causes", "recommended_next_steps",
                "appeal_checklist", "risk_warnings", "missing_information_needed"]:
        text_parts.extend(data.get(key, []))

    full_text = " ".join(str(p) for p in text_parts).lower()

    for kw in keywords:
        if kw.lower() in full_text:
            return True, f"found: {kw}"

    return False, f"none of {keywords} found"

def check_must_not_include(data, expectations):
    """Check that forbidden content does NOT appear in the response."""
    forbidden = expectations.get("must_not_include", [])
    if not forbidden:
        return True, "no must_not_include set"

    text_parts = []
    text_parts.append(data.get("plain_english_explanation", ""))
    for key in ["likely_root_causes", "recommended_next_steps",
                "appeal_checklist", "risk_warnings", "missing_information_needed"]:
        text_parts.extend(data.get(key, []))

    full_text = " ".join(str(p) for p in text_parts).lower()

    found = []
    for term in forbidden:
        if term.lower() in full_text:
            found.append(term)

    if found:
        return False, f"forbidden content found: {found}"
    return True, "no forbidden content"


def check_missing_info(data, expectations):
    """Check missing_information_needed meets expectations."""
    actual_missing = data.get("missing_information_needed") or []

    # Check minimum count
    min_count = expectations.get("expected_missing_info_min")
    if min_count is not None:
        if len(actual_missing) < min_count:
            return False, f"expected >= {min_count} missing items, got {len(actual_missing)}"

    # Check exact expected items (if specified as a list)
    expected_items = expectations.get("expected_missing_info")
    if expected_items is not None:
        if isinstance(expected_items, list) and len(expected_items) == 0:
            # Expectation: no missing info
            if len(actual_missing) > 0:
                return False, f"expected no missing info, got {len(actual_missing)}"

    return True, f"missing info count: {len(actual_missing)}"


def check_next_steps_count(data, expectations):
    """Check recommended_next_steps meets minimum count."""
    min_steps = expectations.get("min_next_steps")
    if min_steps is None:
        return True, "no min_next_steps set"

    actual = len(data.get("recommended_next_steps") or [])
    if actual < min_steps:
        return False, f"expected >= {min_steps} next steps, got {actual}"
    return True, f"next steps count: {actual}"


def check_root_causes_count(data, expectations):
    """Check likely_root_causes meets minimum count."""
    min_causes = expectations.get("min_root_causes")
    if min_causes is None:
        return True, "no min_root_causes set"

    actual = len(data.get("likely_root_causes") or [])
    if actual < min_causes:
        return False, f"expected >= {min_causes} root causes, got {actual}"
    return True, f"root causes count: {actual}"


def run_all_checks(data, expectations):
    """Run all evaluation checks. Returns dict of check_name -> (passed, detail)."""
    results = {}
    results["schema_valid"] = check_schema_valid(data)
    results["confidence_band"] = check_confidence_band(data, expectations)
    results["must_include"] = check_must_include(data, expectations)
    results["must_not_include"] = check_must_not_include(data, expectations)
    results["must_include_any"] = check_must_include_any(data, expectations)
    results["missing_info"] = check_missing_info(data, expectations)
    results["next_steps_count"] = check_next_steps_count(data, expectations)
    results["root_causes_count"] = check_root_causes_count(data, expectations)
    return results


# ---------------------------------------------------------------------------
# Consistency check across multiple runs
# ---------------------------------------------------------------------------
def check_consistency(all_runs_data):
    """
    Compare results across multiple runs of the same case.
    Returns (is_consistent, detail_dict).
    """
    if len(all_runs_data) < 2:
        return True, {"note": "only 1 run, skipping consistency"}

    confidences = [d.get("confidence") for d in all_runs_data]
    confidence_consistent = len(set(confidences)) == 1

    # Check if root causes overlap significantly
    all_causes = []
    for d in all_runs_data:
        causes = set(c.lower().strip() for c in (d.get("likely_root_causes") or []))
        all_causes.append(causes)

    # Jaccard similarity between first run and others
    if all_causes[0]:
        overlaps = []
        for other in all_causes[1:]:
            if all_causes[0] or other:
                union = all_causes[0] | other
                intersection = all_causes[0] & other
                jaccard = len(intersection) / len(union) if union else 1.0
                overlaps.append(jaccard)
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 1.0
    else:
        avg_overlap = 1.0

    is_consistent = confidence_consistent and avg_overlap >= 0.3

    return is_consistent, {
        "confidence_values": confidences,
        "confidence_consistent": confidence_consistent,
        "root_cause_jaccard_avg": round(avg_overlap, 3),
    }


# ---------------------------------------------------------------------------
# Run a single test case
# ---------------------------------------------------------------------------
def run_single_case(client, case, num_runs=3):
    """
    Run a single test case num_runs times.
    Returns a result dict with per-run details and consistency info.
    """
    test_id = case["test_id"]
    category = case.get("category", "unknown")
    description = case.get("description", "")
    denial_input = case["input"]
    expectations = case.get("expectations", {})

    print(f"  [{test_id}] {description}")

    runs = []
    all_data = []

    for run_num in range(1, num_runs + 1):
        run_result = {
            "run_number": run_num,
            "error": None,
            "truncated": False,
            "data": None,
            "checks": {},
            "tokens": {"input": 0, "output": 0},
            "cost_estimate": 0.0,
        }

        try:
            enriched = enrich_input(denial_input)
            data = run_denial_explainer(
                client,
                model=MODEL,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(enriched),
                max_tokens=MAX_TOKENS,
                prompt_version=PROMPT_VERSION,
                temperature=TEMPERATURE,
            )

            # Extract meta
            meta = data.pop("_meta", {})
            usage = meta.get("usage", {})
            in_tok = usage.get("input_tokens", 0)
            out_tok = usage.get("output_tokens", 0)

            run_result["data"] = data
            run_result["tokens"] = {"input": in_tok, "output": out_tok}
            # Estimate cost (Sonnet 4.5: $3/MTok input, $15/MTok output)
            run_result["cost_estimate"] = round(
                (in_tok * 3 + out_tok * 15) / 1_000_000, 6
            )

            # Run checks
            run_result["checks"] = run_all_checks(data, expectations)
            all_data.append(data)

            # Include actual response data for failed checks debugging
            run_result["actual_output"] = {
                "confidence": data.get("confidence"),
                "plain_english_explanation": data.get("plain_english_explanation"),
                "likely_root_causes": data.get("likely_root_causes"),
                "missing_information_needed": data.get("missing_information_needed"),
                "recommended_next_steps": data.get("recommended_next_steps"),
                "risk_warnings": data.get("risk_warnings"),
            }

        except Exception as e:
            err_str = str(e)
            run_result["error"] = err_str[:300]
            if "truncat" in err_str.lower() or "brace" in err_str.lower():
                run_result["truncated"] = True

            # Still record a failed checks set
            run_result["checks"] = {
                "schema_valid": (False, f"error: {err_str[:100]}"),
                "confidence_band": (False, "no data"),
                "must_include": (False, "no data"),
                "must_not_include": (True, "no data to check"),
                "missing_info": (False, "no data"),
                "next_steps_count": (False, "no data"),
                "root_causes_count": (False, "no data"),
            }

        runs.append(run_result)

        # Brief pause between runs to avoid rate limiting
        if run_num < num_runs:
            time.sleep(1)

    # Consistency across runs
    consistency_passed, consistency_detail = check_consistency(all_data)

    # Aggregate pass/fail
    total_checks = 0
    total_passed = 0
    for run in runs:
        for check_name, (passed, _) in run["checks"].items():
            total_checks += 1
            if passed:
                total_passed += 1

    any_truncated = any(r["truncated"] for r in runs)
    any_error = any(r["error"] is not None for r in runs)
    all_schema_valid = all(
        r["checks"].get("schema_valid", (False,))[0] for r in runs
    )

    # Overall pass: all checks passed across all runs AND consistent
    overall_pass = (total_passed == total_checks) and consistency_passed and not any_error

    # Cost totals
    total_cost = sum(r["cost_estimate"] for r in runs)
    total_input_tokens = sum(r["tokens"]["input"] for r in runs)
    total_output_tokens = sum(r["tokens"]["output"] for r in runs)

    # Status indicator for console
    if overall_pass:
        status = "PASS"
    elif any_error:
        status = "ERROR"
    else:
        status = "FAIL"

    print(f"         {status} ({total_passed}/{total_checks} checks, consistency={consistency_passed})")

    return {
        "test_id": test_id,
        "category": category,
        "description": description,
        "status": status,
        "overall_pass": overall_pass,
        "num_runs": num_runs,
        "runs": runs,
        "consistency": {
            "passed": consistency_passed,
            "detail": consistency_detail,
        },
        "summary": {
            "checks_passed": total_passed,
            "checks_total": total_checks,
            "any_truncated": any_truncated,
            "any_error": any_error,
            "all_schema_valid": all_schema_valid,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_cost_estimate": round(total_cost, 6),
        },
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_reports(all_results, prompt_version, num_runs):
    """Generate JSON report, CSV summary, and console summary."""
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"eval_{prompt_version}_{timestamp}"

    # --- JSON report (detailed) ---
    json_path = EVAL_OUTPUT_DIR / f"{base_name}.json"
    report = {
        "metadata": {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "prompt_version": prompt_version,
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "runs_per_case": num_runs,
            "total_cases": len(all_results),
        },
        "results": all_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # --- CSV summary (one row per case) ---
    csv_path = EVAL_OUTPUT_DIR / f"{base_name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "test_id",
            "category",
            "status",
            "checks_passed",
            "checks_total",
            "consistency",
            "any_truncated",
            "any_error",
            "all_schema_valid",
            "total_input_tokens",
            "total_output_tokens",
            "total_cost",
            "description",
        ])

        for r in all_results:
            s = r["summary"]
            writer.writerow([
                r["test_id"],
                r["category"],
                r["status"],
                s["checks_passed"],
                s["checks_total"],
                r["consistency"]["passed"],
                s["any_truncated"],
                s["any_error"],
                s["all_schema_valid"],
                s["total_input_tokens"],
                s["total_output_tokens"],
                s["total_cost_estimate"],
                r["description"],
            ])

    # --- Console summary ---
    print("\n")
    print("=" * 65)
    print("  EVALUATION REPORT")
    print("=" * 65)
    print(f"  Prompt version : {prompt_version}")
    print(f"  Model          : {MODEL}")
    print(f"  Runs per case  : {num_runs}")
    print(f"  Total cases    : {len(all_results)}")
    print("-" * 65)

    passed = sum(1 for r in all_results if r["overall_pass"])
    failed = sum(1 for r in all_results if r["status"] == "FAIL")
    errored = sum(1 for r in all_results if r["status"] == "ERROR")
    total = len(all_results)

    total_cost = sum(r["summary"]["total_cost_estimate"] for r in all_results)
    total_input = sum(r["summary"]["total_input_tokens"] for r in all_results)
    total_output = sum(r["summary"]["total_output_tokens"] for r in all_results)

    print(f"  PASSED  : {passed}/{total}")
    print(f"  FAILED  : {failed}/{total}")
    print(f"  ERRORS  : {errored}/{total}")
    print(f"  Tokens  : {total_input} in / {total_output} out")
    print(f"  Est cost: ${total_cost:.4f}")
    print("-" * 65)

    # Show failures
    failures = [r for r in all_results if not r["overall_pass"]]
    if failures:
        print("  FAILURES:")
        for r in failures:
            print(f"    [{r['test_id']}] {r['status']} - {r['description']}")

            if r["runs"]:
                first_run = r["runs"][0]
                if first_run.get("error"):
                    print(f"           error: {first_run['error'][:100]}")
                for check_name, (check_passed, detail) in first_run["checks"].items():
                    if not check_passed:
                        print(f"           FAILED {check_name}: {detail}")

                # Show actual output for context
                actual = first_run.get("actual_output")
                if actual:
                    print(f"           --- actual output ---")
                    print(f"           confidence: {actual.get('confidence')}")
                    expl = actual.get('plain_english_explanation', '')
                    print(f"           explanation: {expl[:120]}")
                    missing = actual.get('missing_information_needed', [])
                    if missing:
                        print(f"           missing_info: {missing}")
                    risks = actual.get('risk_warnings', [])
                    if risks:
                        print(f"           risk_warnings: {risks}")

            if not r["consistency"]["passed"]:
                print(f"           consistency: {r['consistency']['detail']}")

            print()

        print("-" * 65)

    # Summary line (the one you screenshot for LinkedIn)
    summary_line = f"{passed}/{total} passed"
    if failed:
        summary_line += f", {failed} failures"
    if errored:
        summary_line += f", {errored} errors"
    print(f"\n  >>> {summary_line}")
    print(f"  >>> Total cost: ${total_cost:.4f}")
    print()

    print(f"  Detailed report : {json_path}")
    print(f"  CSV summary     : {csv_path}")
    print("=" * 65)

    return json_path, csv_path


# ---------------------------------------------------------------------------
# Dry run (validate case files only)
# ---------------------------------------------------------------------------
def dry_run(cases):
    """Validate all case files without calling the API."""
    print("\n  DRY RUN - Validating case files\n")
    errors_found = 0

    for case in cases:
        filepath = case.get("_filepath", "unknown")
        file_errors = validate_case_file(case)

        if file_errors:
            print(f"  INVALID [{case.get('test_id', '???')}] {filepath}")
            for e in file_errors:
                print(f"           - {e}")
            errors_found += 1
        else:
            print(f"  OK      [{case.get('test_id', '???')}] {case.get('description', '')}")

    print(f"\n  {len(cases)} cases checked, {errors_found} with errors.\n")
    return errors_found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Denial Explainer Evaluation Runner")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"Number of runs per case (default: {DEFAULT_RUNS})")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter: 'standard', 'adversarial', or a specific category name")
    parser.add_argument("--case", type=str, default=None,
                        help="Run a single case by test_id (e.g., '001')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate case files without calling the API")
    args = parser.parse_args()

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY not set.")
        sys.exit(1)

    # Load cases
    cases = load_all_cases(category_filter=args.category, case_filter=args.case)

    if not cases:
        print("No eval cases found. Check eval_cases/ folder.")
        print(f"  Looking in: {EVAL_CASES_DIR}")
        sys.exit(1)

    print(f"\n  Loaded {len(cases)} eval case(s)")

    # Dry run mode
    if args.dry_run:
        errors = dry_run(cases)
        sys.exit(1 if errors else 0)

    # Estimate cost before running
    estimated_calls = len(cases) * args.runs
    # Rough estimate: ~600 input + ~500 output per call at Sonnet pricing
    rough_cost = estimated_calls * (600 * 3 + 500 * 15) / 1_000_000
    print(f"  Estimated API calls: {estimated_calls}")
    print(f"  Estimated cost: ~${rough_cost:.4f}")
    print(f"  Runs per case: {args.runs}")
    print()

    # Confirm if many calls
    if estimated_calls > 20:
        confirm = input(f"  This will make {estimated_calls} API calls. Continue? [y/N] ")
        if confirm.lower() != "y":
            print("  Aborted.")
            sys.exit(0)

    # Run
    client = Anthropic(api_key=api_key)
    all_results = []

    print("\n  Running evaluations...\n")

    for case in cases:
        result = run_single_case(client, case, num_runs=args.runs)
        all_results.append(result)

    # Generate reports
    generate_reports(all_results, PROMPT_VERSION, args.runs)


if __name__ == "__main__":
    main()