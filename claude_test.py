import os
from anthropic import Anthropic

MODEL = "claude-sonnet-4-5"

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

response = client.messages.create(
    model=MODEL,
    max_tokens=150,
    messages=[
        {"role": "user", "content": "Explain insurance claim denial in simple words."}
    ],
)

print("\nClaude says:\n")
print(response.content[0].text)
