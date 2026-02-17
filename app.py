import json
import os
import streamlit as st
from anthropic import Anthropic

# Importing existing functions from denial_explainer.py
from denial_explainer import build_user_prompt, extract_json, SYSTEM_PROMPT, MODEL

st.set_page_config(page_title="Denial Explainer MVP", layout="wide")
st.title("🧾 Denial Explainer (MVP)")
st.info("⚠️ Keep responses concise to control API costs.")
st.caption("Paste a denial input JSON → get structured explanation + appeal guidance (Claude).")

# --- API key check ---
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    st.error("ANTHROPIC_API_KEY is not set. Set it in environment variables and restart VS Code.")
    st.stop()

client = Anthropic(api_key=api_key)

# --- UI layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input (Denial JSON)")
    default_input = {
        "payer": "Example Health Plan",
        "denial_code_or_reason": "Service not medically necessary",
        "member_context": "Outpatient imaging for back pain",
        "provider_context": "Ordering physician submitted clinical notes",
        "dates": {"service_date": "2026-02-01"},
    }
    input_text = st.text_area(
        "Edit JSON below:",
        value=json.dumps(default_input, indent=2),
        height=350
    )

    max_tokens = st.slider("Max tokens (cost control)", 200, 1000, 700, 50)
    st.caption("Tip: keep max_tokens 400–800 for cheap, fast responses.")


 #   max_tokens = st.slider("Max tokens (cost control)", min_value=200, max_value=1200, value=900, step=50)

    run_btn = st.button("Run Claude → Explain Denial", type="primary")

with col2:
    st.subheader("Output (Structured JSON)")
    output_box = st.empty()

# --- Run logic ---
if run_btn:
    # Validate input JSON
    try:
        denial_input = json.loads(input_text)
    except json.JSONDecodeError as e:
        output_box.error(f"Invalid input JSON: {e}")
        st.stop()

    with st.spinner("Calling Claude..."):
        
        resp = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(denial_input)}],
        )

        # Extract text blocks
        text_parts = []
        for block in resp.content:
            txt = getattr(block, "text", None)
            if isinstance(txt, str) and txt.strip():
                text_parts.append(txt)
        text = "\n".join(text_parts).strip()

        if "}" not in text:
            output_box.error("Output looks truncated. Increase max_tokens or shorten constraints.")
            st.code(text)
            st.stop()

        json_text = extract_json(text)
        if not json_text:
            output_box.error("Could not find JSON in model output.")
            st.code(text)
            st.stop()

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            output_box.error(f"Claude returned invalid JSON: {e}")
            st.code(text)
            st.stop()

        output_box.json(data)
        st.success("Response generated successfully")
        confidence = data.get("confidence", "").lower()

        if confidence == "high":
            st.success("Confidence: HIGH")
        elif confidence == "medium":
            st.warning("Confidence: MEDIUM")
        else:
            st.error("Confidence: LOW — more information may be needed")

    st.download_button(
        label="📥 Download JSON",
        data=json.dumps(data, indent=2),
        file_name="denial_explanation.json",
        mime="application/json"
    )
    st.text_area(
    "Copy JSON",
    value=json.dumps(data, indent=2),
    height=200
    )

