import os
from anthropic import Anthropic

api_key = os.getenv("ANTHROPIC_API_KEY")

if not api_key:
    raise SystemExit("API key not found.")

client = Anthropic(api_key=api_key)

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Explain Healthcare payer insurance claim denial in simple words."}
    ],
)

print("\nClaude says:\n")
print(response.content[0].text)
