import requests

response = requests.get("https://api.github.com")

data = response.json()

print("Status:", response.status_code)
print("GitHub API rate limit URL:", data["rate_limit_url"])
