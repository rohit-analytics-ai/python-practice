import json
import os
import streamlit as st
from anthropic import Anthropic

def classify_severity(data):
    explanation = data.get("plain_english_explanation", "").lower()
    risks = " ".join(data.get("risk_warnings", [])).lower()

    if "full cost" in risks or "responsible" in risks:
        return "HIGH"
    if "appeal" in explanation or "denied" in explanation:
        return "MEDIUM"
    return "LOW"


def generate_appeal_text(data):
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



# Reuse logic from your CLI tool
from denial_explainer import build_user_prompt, extract_json, SYSTEM_PROMPT, MODEL

st.set_page_config(page_title="Denial Explainer MVP", layout="wide")
st.title("🧾 Denial Explainer (MVP)")
st.caption("Paste a denial input JSON → get structured explanation + appeal guidance (Claude).")
st.caption("⚠️ Always verify guidance against payer policy and clinical documentation.")

# --- Session history (keeps last runs during this browser session) ---
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"input":..., "output":...}

# --- API key check ---
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("ANTHROPIC_API_KEY is not set. Set it in your environment variables and restart VS Code.")
    st.stop()

client = Anthropic(api_key=api_key)

# --- Mode toggle ---
compare_mode = st.checkbox("Compare two denials side-by-side", value=False)

# --- UI layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input (Denial JSON)")

    default_input_1 = {
        "payer": "Example Health Plan",
        "denial_code_or_reason": "Service not medically necessary",
        "member_context": "Outpatient imaging for back pain",
        "provider_context": "Ordering physician submitted clinical notes",
        "dates": {"service_date": "2026-02-01"},
    }

    default_input_2 = {
        "payer": "Example Health Plan",
        "denial_code_or_reason": "Prior authorization required",
        "member_context": "Outpatient MRI",
        "provider_context": "Radiology facility billed without auth on file",
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
            height=350
        )

    max_tokens = st.slider("Max tokens (cost control)", min_value=200, max_value=1000, value=700, step=50)
    st.caption("Tip: keep max_tokens 400–800 for cheaper, faster responses.")

    run_btn = st.button("Run Claude", type="primary")

with col2:
    st.subheader("Output")
    output_box = st.empty()


def run_one(denial_payload: dict) -> dict:
    """Call Claude and return parsed JSON dict."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(denial_payload)}],
    )

    # Robust extraction across SDK versions: take any block that has .text
    text_parts = []
    for block in resp.content:
        txt = getattr(block, "text", None)
        if isinstance(txt, str) and txt.strip():
            text_parts.append(txt)

    text = "\n".join(text_parts).strip()

    if "cannot determine" in text.lower() or "insufficient" in text.lower():
        st.error("Claude indicates insufficient information to provide reliable guidance.")
        st.stop()


    if "}" not in text:
        raise ValueError("Output looks truncated. Increase max_tokens or shorten constraints.")

    json_text = extract_json(text)
    if not json_text:
        raise ValueError("Could not find JSON in model output.")

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Claude returned invalid JSON: {e}")


if run_btn:
    # Parse inputs
    if compare_mode:
        try:
            denial_a = json.loads(input_text_a)
            denial_b = json.loads(input_text_b)
        except json.JSONDecodeError as e:
            output_box.error(f"Invalid input JSON: {e}")
            st.stop()
    else:
        try:
            denial_input = json.loads(input_text)
        except json.JSONDecodeError as e:
            output_box.error(f"Invalid input JSON: {e}")
            st.stop()

    with st.spinner("Calling Claude..."):
        if compare_mode:
            try:
                data_a = run_one(denial_a)
                data_b = run_one(denial_b)
            except Exception as e:
                st.error(str(e))
                st.stop()

            # Side-by-side outputs
            out1, out2 = st.columns(2)
            with out1:
                st.subheader("Output — Denial A")
                st.json(data_a)
            with out2:
                st.subheader("Output — Denial B")
                st.json(data_b)

            st.subheader("🔍 Quick Compare")
            st.write("**Confidence:**", data_a.get("confidence"), "vs", data_b.get("confidence"))
            st.write(
                "**Missing info count:**",
                len(data_a.get("missing_information_needed", []) or []),
                "vs",
                len(data_b.get("missing_information_needed", []) or []),
            )

            # Save to history (keep last 5)
            st.session_state.history.insert(0, {"input": {"A": denial_a, "B": denial_b}, "output": {"A": data_a, "B": data_b}})
            st.session_state.history = st.session_state.history[:5]

        else:
            try:
                data = run_one(denial_input)
            except Exception as e:
                st.error(str(e))
                st.stop()

            output_box.json(data)
            st.success("Response generated successfully")

            confidence = (data.get("confidence") or "").lower()
            if confidence == "high":
                st.success("✅ Confidence: HIGH")
            elif confidence == "medium":
                st.warning("⚠️ Confidence: MEDIUM")
            else:
                st.error("❗ Confidence: LOW (needs more info)")

            missing = data.get("missing_information_needed", []) or []
            if missing and confidence == "high":
                st.warning("⚠️ Confidence adjusted: missing information detected")


            severity = classify_severity(data)
            if severity == "HIGH":
                st.error("🚨 Severity: HIGH — financial risk likely")
            elif severity == "MEDIUM":
                st.warning("⚠️ Severity: MEDIUM — review recommended")
            else:
                st.success("✅ Severity: LOW")


            # Missing info panel
            missing = data.get("missing_information_needed", []) or []
            st.subheader("🧩 Missing Information Needed")
            if missing:
                st.warning("Claude says more info is needed to increase confidence:")
                for m in missing:
                    st.write(f"- {m}")
            else:
                st.success("No missing information flagged.")


            st.subheader("📝 Appeal Letter Draft")

            appeal_text = generate_appeal_text(data)

            st.text_area("Editable draft:", value=appeal_text, height=220)

            st.download_button(
                label="📥 Download Appeal Letter",
                data=appeal_text,
                file_name="appeal_letter.txt"
            )


            # Download + Copy
            st.download_button(
                label="📥 Download JSON",
                data=json.dumps(data, indent=2),
                file_name="denial_explanation.json",
                mime="application/json"
            )
            st.text_area("Copy JSON", value=json.dumps(data, indent=2), height=200)

            # Save to history (keep last 5)
            st.session_state.history.insert(0, {"input": denial_input, "output": data})
            st.session_state.history = st.session_state.history[:5]


st.divider()
st.subheader("🕘 History (last 5 runs)")

if not st.session_state.history:
    st.caption("No history yet. Run the tool to see entries here.")
else:
    for i, item in enumerate(st.session_state.history, start=1):
        # Handle both single and compare entries
        if isinstance(item["output"], dict) and "A" in item["output"] and "B" in item["output"]:
            conf_a = (item["output"]["A"].get("confidence") or "").upper()
            conf_b = (item["output"]["B"].get("confidence") or "").upper()
            title = f"Run #{i} — A:{conf_a} / B:{conf_b}"
        else:
            conf = (item["output"].get("confidence") or "").upper()
            title = f"Run #{i} — Confidence: {conf}"

        with st.expander(title):
            st.markdown("**Input**")
            st.json(item["input"])
            st.markdown("**Output**")
            st.json(item["output"])
