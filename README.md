# Denial Explainer (MVP)

A small Claude-powered tool that converts a denial reason into a structured, member-friendly explanation and appeal guidance.

## What it does
- Produces a plain-English explanation of the denial
- Lists likely root causes and missing information needed
- Suggests next steps + an appeal checklist
- Includes risk warnings and a confidence level

## Safety / Accuracy Notes
- Designed to be conservative: if inputs are incomplete, it asks for missing details.
- Does not invent policy specifics or clinical rules.
- Intended for educational / workflow support only (not medical or legal advice).

## Setup (Windows)

```bash
# Create and activate venv
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
