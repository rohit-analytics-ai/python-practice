# app.py
# Streamlit UI for the Denial Explainer MVP
# Notes:
# - Uses core.py for reliable execution (retry + JSON parsing + schema validation)
# - Keeps UI logic here and shared "engine" logic in core.py

import json
import os
import datetime
from pathlib import Path

import streamlit as st
from anthropic import Anthropic

from denial_explainer import build_user_prompt, SYSTEM_PROMPT, MODEL, PROMPT_VERSION
#from denial_explainer import build_user_prompt, build_user_prompt_loose, SYSTEM_PROMPT, MODEL, PROMPT_VERSION

from core import validate_input, run_denial_explainer
from denial_explainer import build_user_prompt, SYSTEM_PROMPT, MODEL, PROMPT_VERSION
from code_lookup import enrich_input, get_sources_used, lookup_carc, lookup_rarc


def friendly_error_message(err: Exception) -> tuple[str, str | None]:
    """
    Returns (user_message, run_id).
    Keeps UI friendly but still shows run_id for debugging.
    """
    msg = str(err)
    run_id = None

    # Try to extract run_id from the error text (we include run_id=... in core.py errors)
    if "run_id=" in msg:
        run_id = msg.split("run_id=", 1)[1].split()[0].strip().strip(":,;")

    # Taxonomy mapping (match the prefixes used in core.py)
    if "ERR_INPUT_VALIDATION" in msg:
        return ("Input looks incomplete/invalid. Please check required fields and try again.", run_id)
    if "ERR_MODEL_TRUNCATED" in msg:
        return ("Model response was cut off. Please try again (or lower output verbosity).", run_id)
    if "ERR_MODEL_NO_JSON" in msg:
        return ("Model did not return valid JSON. Please try again.", run_id)
    if "ERR_MODEL_JSON_PARSE" in msg:
        return ("Model returned malformed JSON. Please try again.", run_id)
    if "ERR_MODEL_SCHEMA_INVALID" in msg:
        return ("Model output didn’t match the expected schema. Please try again.", run_id)
    if "ERR_API" in msg or "API" in msg:
        return ("API call failed or timed out. Please try again.", run_id)

    # Fallback
    return ("Something went wrong generating the explanation. Please try again.", run_id)



def classify_severity(data: dict) -> str:
    """
    Lightweight severity indicator.
    This is a heuristic (string matching) and will be improved later with a rubric + evaluation.
    """
    explanation = (data.get("plain_english_explanation") or "").lower()
    risks = " ".join(data.get("risk_warnings") or []).lower()

    if "full cost" in risks or "responsible" in risks:
        return "HIGH"
    if "appeal" in explanation or "denied" in explanation:
        return "MEDIUM"
    return "LOW"


def generate_appeal_text(data: dict) -> str:
    """
    Generates a simple appeal letter draft based on model output.
    Keeps it editable for the user (never claims it is legally/clinically authoritative).
    """
    explanation = data.get("plain_english_explanation", "")
    causes = data.get("likely_root_causes", [])
    missing = data.get("missing_information_needed", [])

    text = f"""To Whom It May Concern,

I am requesting reconsideration of the denial for the referenced service.

Reason provided: {explanation}

Potential issues identified:
- """ + "\n- ".join(causes)

    if missing:
        text += "\n\nAdditional documentation can be provided if needed, including:\n- " + "\n- ".join(missing)

    text += "\n\nPlease review this request and advise if further information is required.\n\nSincerely,\nMember/Provider"
    return text


def render_feedback(last_result: dict) -> None:
    """Save thumbs up/down feedback to feedback/feedback.jsonl."""
    if not last_result:
        return

    meta = last_result["meta"]
    data = last_result["data"]
    in_tok = last_result["in_tok"]
    out_tok = last_result["out_tok"]

    st.subheader("Feedback")

    base_dir = Path(__file__).resolve().parent
    feedback_path = base_dir / "feedback" / "feedback.jsonl"
    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    st.caption(f"Feedback file: {feedback_path}")
 #   st.caption(f"Exists right now: {feedback_path.exists()}")

    def write_feedback(label: str) -> None:
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "run_id": meta.get("run_id"),
            "label": label,
            "confidence": data.get("confidence") if isinstance(data, dict) and "confidence" in data else None,
            "model": meta.get("model"),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "prompt_version": meta.get("prompt_version"),
        }
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())
        # st.info(f"Wrote 1 line. File exists now: {feedback_path.exists()}")

        # Debug: show the latest line saved (proves the write happened)
        with open(feedback_path, "r", encoding="utf-8") as rf:
            lines = rf.readlines()
            st.code(lines[-1] if lines else "(file empty)")


    col_fb1, col_fb2 = st.columns(2)
    run_id = meta.get("run_id") or "no_run_id"
    
    st.caption(f"Feedback for Run ID: {meta.get('run_id') or 'no_run_id'}")

    with col_fb1:
        if st.button("👍Accurate", key=f"fb_up_{run_id}"):
            write_feedback("up")
            st.success(f"Saved to {feedback_path}")

    with col_fb2:
        if st.button("👎Not accurate", key=f"fb_down_{run_id}"):
            write_feedback("down")
            st.success(f"Saved to {feedback_path}")


# ---------- Page setup ----------
st.set_page_config(page_title="Denial Explainer MVP", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; font-size: 42px; margin-bottom: 0;'>
    Denial Explainer
    </h1>
    <p style='text-align: center; font-size: 18px; color: #555; margin-top: 4px;'>
    Turn claim denials into clear explanations, appeal guidance, and action steps.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="
        background-color:#f3f6fb;
        padding:12px 18px;
        border-radius:10px;
        text-align:center;
        font-size:14px;
        color:#444;
        margin-top:10px;
        margin-bottom:20px;
    ">
    Guidance is educational. Always verify against payer policy and clinical documentation.
    </div>
    """,
    unsafe_allow_html=True,
)

# Keep history during this browser session
if "history" not in st.session_state:
    st.session_state.history = []

# Persist last output so the UI doesn't go blank on reruns (e.g., feedback clicks)
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "temp_results" not in st.session_state:
    st.session_state.temp_results = None
# ---------- API key ----------
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("ANTHROPIC_API_KEY is not set. Set it in your environment variables and restart.")
    st.stop()

client = Anthropic(api_key=api_key)

# ---------- Mode toggle ----------
compare_mode = st.checkbox("Compare two denials side-by-side", value=False)

#----------Temp-experiment only for Normal mode
if compare_mode:
    temp_experiment = False
else:
    temp_experiment = st.checkbox(
        "Run temperature experiment (0.0, 0.3, 0.7)"
    )

# ---------- UI layout ----------
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div style="
            background-color:#eef2f7;
            padding:10px 14px;
            border-radius:8px;
            font-weight:600;
            font-size:16px;
        ">
        Input (Denial JSON)
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Guidance for better accuracy & compliance
    st.info(
        "For best accuracy, include CARC/RARC codes from your ERA/EOB, CPT codes, "
        "and ICD-10 diagnosis codes. Do not include patient-identifying information."
    )

    default_input_1 = {
        "payer": "Example Health Plan",
        "carc_code": "CO-50",
        "rarc_code": "N579",
        "denial_reason_text": "Non-covered service, not deemed medically necessary",
        "member_context": "Outpatient MRI of lumbar spine for chronic low back pain",
        "provider_context": "Ordering physician submitted clinical notes and prior conservative treatment records",
        "claim_type": "Outpatient",
        "codes": ["72148"],
        "diagnosis_codes": ["M54.5"],
        "place_of_service": "22",
        "dates": {"service_date": "2026-01-25"},
    }

    default_input_2 = {
        "payer": "Example Health Plan",
        "carc_code": "CO-15",
        "rarc_code": "MA01",
        "denial_reason_text": "Authorization required but not obtained prior to service",
        "member_context": "Outpatient MRI of knee",
        "provider_context": "Radiology facility billed without auth on file",
        "claim_type": "Outpatient",
        "codes": ["73721"],
        "diagnosis_codes": ["M23.50"],
        "place_of_service": "22",
        "dates": {"service_date": "2026-01-18"},
    }


    if compare_mode:
        st.markdown("**Denial A**")
        input_text_a = st.text_area(
            "A",
            value=json.dumps(default_input_1, indent=2),
            height=250,
            label_visibility="collapsed",
        )

        st.markdown("**Denial B**")
        input_text_b = st.text_area(
            "B",
            value=json.dumps(default_input_2, indent=2),
            height=250,
            label_visibility="collapsed",
        )
    else:
        input_text = st.text_area(
            "Edit JSON below:",
            value=json.dumps(default_input_1, indent=2),
            height=350,
        )

    max_tokens = st.slider("Max tokens (cost control)", 200, 1000, 700, 50)
    st.caption("Tip: keep max_tokens 400-800 for cheaper, faster responses.")

    run_btn = st.button("Run Claude", type="primary")

with col2:
    st.subheader("Output")

    # Show persisted result in the Output column (survives reruns)
    if st.session_state.last_result:
        last = st.session_state.last_result
        if last["mode"] == "single":
            st.json(last["data"])
        else:
            sub_a, sub_b = st.columns(2)
            with sub_a:
                st.markdown("**Denial A**")
                st.json(last["data"]["A"])
            with sub_b:
                st.markdown("**Denial B**")
                st.json(last["data"]["B"])

        st.caption(f"Run ID: {last['meta'].get('run_id')}")
        st.caption(
            f"Tokens -- input: {last['in_tok']}, output: {last['out_tok']}, "
            f"total: {last['in_tok'] + last['out_tok']}"
        )
    else:
        st.info("Click 'Run Claude' to see results here.")


# ---------- Run ----------
if run_btn:
    # Parse user inputs from text areas
    if compare_mode:
        try:
            denial_a = json.loads(input_text_a)
            denial_b = json.loads(input_text_b)
        except json.JSONDecodeError as e:
            st.error(f"Invalid input JSON: {e}")
            st.stop()
    else:
        try:
            denial_input = json.loads(input_text)
        except json.JSONDecodeError as e:
            st.error(f"Invalid input JSON: {e}")
            st.stop()

    with st.spinner("Calling Claude..."):
        if compare_mode:
            errors_a = validate_input(denial_a)
            errors_b = validate_input(denial_b)
            if errors_a or errors_b:
                st.error(f"INPUT_INVALID: A={errors_a} B={errors_b}")
                st.stop()

            try:
                enriched_a = enrich_input(denial_a)
                enriched_b = enrich_input(denial_b)
                data_a = run_denial_explainer(
                    client,
                    model=MODEL,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=build_user_prompt(enriched_a),
                    max_tokens=max_tokens,
                    prompt_version=PROMPT_VERSION,
                    temperature=0.0,
                )
                data_b = run_denial_explainer(
                    client,
                    model=MODEL,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=build_user_prompt(enriched_b),
                    max_tokens=max_tokens,
                    prompt_version=PROMPT_VERSION,
                    temperature=0.0,
                )
            except Exception as e:
                user_msg, run_id = friendly_error_message(e)
                st.error(user_msg)
                if run_id:
                    st.caption(f"run_id: {run_id}")
                # Optional: show short debug detail without dumping everything
                st.caption(f"details: {str(e)[:200]}")
                st.stop()

            meta = data_a.get("_meta", {})
            usage = meta.get("usage") or {}
            in_tok = usage.get("input_tokens") or 0
            out_tok = usage.get("output_tokens") or 0

            # Save to session_state -- col2 reads from here on rerun
            st.session_state.last_result = {
                "mode": "compare",
                "input": {"A": denial_a, "B": denial_b},
                "enriched_input": {"A": enriched_a, "B": enriched_b},
                "data": {"A": data_a, "B": data_b},
                "meta": meta,
                "in_tok": in_tok,
                "out_tok": out_tok,
            }

            estimated_cost = (in_tok + out_tok) / 1_000_000 * 10
            st.caption(f"Estimated cost: ~${estimated_cost:.4f} (rough estimate)")

            st.subheader("Quick Compare")
            st.write("**Confidence:**", data_a.get("confidence"), "vs", data_b.get("confidence"))
            st.write(
                "**Missing info count:**",
                len(data_a.get("missing_information_needed") or []),
                "vs",
                len(data_b.get("missing_information_needed") or []),
            )

            st.session_state.history.insert(
                0,
                {"input": {"A": denial_a, "B": denial_b}, "output": {"A": data_a, "B": data_b}},
            )
            st.session_state.history = st.session_state.history[:5]

            # Force rerun so col2 picks up the new result
            # Skip rerun for temp experiment so the side-by-side comparison stays visible
            if not temp_experiment:
                st.rerun()

        else:
            errors = validate_input(denial_input)
            if errors:
                st.error(f"INPUT_INVALID: {errors}")
                st.stop()

            try:
                # 🔬 Temperature experiment mode
                if temp_experiment:
                    temps = [0.0, 0.3, 0.7]
                    temp_results = []
                    enriched_input = enrich_input(denial_input)
                    # Use higher max_tokens for temp experiment since higher temps = more verbose
                    exp_max_tokens = min(max_tokens + 300, 1000)

                    for t in temps:
                        try:
                            result = run_denial_explainer(
                                client,
                                model=MODEL,
                                system_prompt=SYSTEM_PROMPT,
                                #user_prompt=build_user_prompt(denial_input),
                                user_prompt=build_user_prompt(enriched_input),
                                max_tokens=exp_max_tokens,
                                prompt_version=PROMPT_VERSION,
                                temperature=t,
                            )
                            temp_results.append({"temp": t, "data": result})
                        except Exception as e:
                            temp_results.append({
                                "temp": t,
                                "data": {"error": str(e)[:150], "confidence": "N/A"}
                            })

                    # Use the temp=0.0 result as the "main" result for session state
                    # Use first successful result as the "main" data
                    data = next(
                        (tr["data"] for tr in temp_results if "error" not in tr["data"]),
                        temp_results[0]["data"]
                    )

                    # Save temp experiment results so persistent section can render them
                    st.session_state.temp_results = temp_results
                    
                  
                    
                else:
                    st.session_state.temp_results = None 
                    enriched_input = enrich_input(denial_input)
                    data = run_denial_explainer(
                        client,
                        model=MODEL,
                        system_prompt=SYSTEM_PROMPT,
                        #user_prompt=build_user_prompt(denial_input),
                        user_prompt=build_user_prompt(enriched_input),
                        max_tokens=max_tokens,
                        prompt_version=PROMPT_VERSION,
                        temperature=0.0,
                    )



######################################################


            except Exception as e:
                user_msg, run_id = friendly_error_message(e)
                st.error(user_msg)
                if run_id:
                    st.caption(f"run_id: {run_id}")
                # Optional: show short debug detail without dumping everything
                st.caption(f"details: {str(e)[:200]}")
                st.stop()

            meta = data.get("_meta", {})
            usage = meta.get("usage") or {}
            in_tok = usage.get("input_tokens") or 0
            out_tok = usage.get("output_tokens") or 0

            # Save to session_state -- col2 reads from here on rerun
            st.session_state.last_result = {
                "mode": "single",
                "input": denial_input,
                "enriched_input": enriched_input,
                "data": data,
                "meta": meta,
                "in_tok": in_tok,
                "out_tok": out_tok,
            }

            st.session_state.history.insert(0, {"input": denial_input, "output": data})
            st.session_state.history = st.session_state.history[:5]

            # Force rerun so col2 picks up the new result
            st.rerun()


# ---------- Details (shown below columns, reads from session_state) ----------
if st.session_state.last_result:
    last = st.session_state.last_result
    st.divider()

    # ---------- Details (shown below columns, reads from session_state) ----------
if st.session_state.last_result:
    last = st.session_state.last_result
    st.divider()

    # Temperature experiment display
    if st.session_state.temp_results:
        st.subheader("Temperature Experiment Results")
        temp_res = st.session_state.temp_results
        tcols = st.columns(len(temp_res))
        for idx, tr in enumerate(temp_res):
            with tcols[idx]:
                st.markdown(f"**Temp = {tr['temp']}**")
                if "error" in tr["data"]:
                    st.error(f"Failed: {tr['data']['error'][:100]}")
                else:
                    st.write(f"**Confidence:** {tr['data'].get('confidence', 'N/A')}")
                    st.write(f"**Explanation:** {tr['data'].get('plain_english_explanation', 'N/A')}")
                    steps = tr["data"].get("recommended_next_steps") or []
                    if steps:
                        st.write("**Next steps:**")
                        for s in steps:
                            st.write(f"- {s}")
                    with st.expander("Full JSON"):
                        st.json(tr["data"])

    elif last["mode"] == "single":

        estimated_cost = (last["in_tok"] + last["out_tok"]) / 1_000_000 * 10
        st.caption(f"Estimated cost: ~${estimated_cost:.4f} (rough estimate)")

        data = last["data"]
        confidence = (data.get("confidence") or "").lower()
        if confidence == "high":
            st.success("Confidence: HIGH")
        elif confidence == "medium":
            st.warning("Confidence: MEDIUM")
        else:
            st.error("Confidence: LOW (needs more info)")

        missing = data.get("missing_information_needed") or []
        if missing and confidence == "high":
            st.warning("Confidence may be overstated because missing information was detected.")

        severity = classify_severity(data)
        if severity == "HIGH":
            st.error("Severity: HIGH -- financial risk likely")
        elif severity == "MEDIUM":
            st.warning("Severity: MEDIUM -- review recommended")
        else:
            st.success("Severity: LOW")

        enrichment = last.get("enriched_input", {}).get("_code_enrichment", {})
        if enrichment:
            st.subheader("Code Lookup (Grounding Sources)")
            carc = enrichment.get("carc_lookup", {})
            if carc.get("found"):
                st.write(f"**CARC {carc['number']}:** {carc['description']}")
                st.write(f"**Category:** {carc['category']}")
                st.write(f"**Appeal guidance:** {carc['appeal_guidance']}")
            else:
                st.warning(f"CARC code not found in lookup table: {carc.get('raw_input', 'N/A')}")
            rarc = enrichment.get("rarc_lookup", {})
            if rarc.get("found"):
                st.write(f"**RARC {rarc['code']}:** {rarc['description']}")
            elif rarc.get("raw_input"):
                st.warning(f"RARC code not found in lookup table: {rarc.get('raw_input', 'N/A')}")

        sources = last.get("data", {}).get("sources_used", [])
        if sources:
            st.subheader("Sources Used by Claude")
            for s in sources:
                st.write(f"- {s}")

                
        st.subheader("Missing Information Needed")
        if missing:
            st.warning("More info is needed to increase confidence:")
            for m in missing:
                st.write(f"- {m}")
        else:
            st.success("No missing information flagged.")

        st.subheader("Appeal Letter Draft")
        appeal_text = generate_appeal_text(data)
        st.text_area("Editable draft:", value=appeal_text, height=220)
        st.download_button("Download Appeal Letter", appeal_text, "appeal_letter.txt")

        st.download_button(
            label="Download JSON",
            data=json.dumps(data, indent=2),
            file_name="denial_explanation.json",
            mime="application/json",
        )
        st.text_area("Copy JSON", value=json.dumps(data, indent=2), height=200)

    else:
        estimated_cost = (last["in_tok"] + last["out_tok"]) / 1_000_000 * 10
        st.caption(f"Estimated cost: ~${estimated_cost:.4f} (rough estimate)")

        st.subheader("Quick Compare")
        st.write("**Confidence:**", last["data"]["A"].get("confidence"), "vs", last["data"]["B"].get("confidence"))
        st.write(
            "**Missing info count:**",
            len(last["data"]["A"].get("missing_information_needed") or []),
            "vs",
            len(last["data"]["B"].get("missing_information_needed") or []),
        )

        enriched_compare = last.get("enriched_input", {})
        if enriched_compare:
            st.subheader("Code Lookup (Grounding Sources)")
            comp_a, comp_b = st.columns(2)
            with comp_a:
                st.markdown("**Denial A**")
                enr_a = enriched_compare.get("A", {}).get("_code_enrichment", {})
                if enr_a:
                    carc = enr_a.get("carc_lookup", {})
                    if carc.get("found"):
                        st.write(f"**CARC {carc['number']}:** {carc['description']}")
                        st.write(f"**Appeal guidance:** {carc['appeal_guidance']}")
                    rarc = enr_a.get("rarc_lookup", {})
                    if rarc.get("found"):
                        st.write(f"**RARC {rarc['code']}:** {rarc['description']}")
            with comp_b:
                st.markdown("**Denial B**")
                enr_b = enriched_compare.get("B", {}).get("_code_enrichment", {})
                if enr_b:
                    carc = enr_b.get("carc_lookup", {})
                    if carc.get("found"):
                        st.write(f"**CARC {carc['number']}:** {carc['description']}")
                        st.write(f"**Appeal guidance:** {carc['appeal_guidance']}")
                    rarc = enr_b.get("rarc_lookup", {})
                    if rarc.get("found"):
                        st.write(f"**RARC {rarc['code']}:** {rarc['description']}")

    # Feedback -- single place, always rendered from session_state
    render_feedback(last)


# ---------- History ----------
st.divider()
st.subheader("History (last 5 runs)")

if not st.session_state.history:
    st.caption("No history yet. Run the tool to see entries here.")
else:
    for i, item in enumerate(st.session_state.history, start=1):
        if isinstance(item["output"], dict) and "A" in item["output"] and "B" in item["output"]:
            conf_a = (item["output"]["A"].get("confidence") or "").upper()
            conf_b = (item["output"]["B"].get("confidence") or "").upper()
            title = f"Run #{i} -- A:{conf_a} / B:{conf_b}"
        else:
            conf = (item["output"].get("confidence") or "").upper()
            title = f"Run #{i} -- Confidence: {conf}"

        with st.expander(title):
            st.markdown("**Input**")
            st.json(item["input"])
            st.markdown("**Output**")
            st.json(item["output"])