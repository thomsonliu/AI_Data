import requests

try:
    response = requests.get(
        "http://localhost:11434/api/tags",
        timeout=5
    )

    response.raise_for_status()

    print("✅ Ollama is running")

except requests.ConnectTimeout:
    print("❌ Could not connect to Ollama")

except requests.RequestException as e:
    print(f"❌ Ollama error: {e}")
