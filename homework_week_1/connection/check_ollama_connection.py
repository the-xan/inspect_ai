import requests


def check_ollama():
    """Check Ollama connection and show installed models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("❌ Ollama returned an error")
            return False

        models = response.json().get('models', [])
        total_size = sum(m['size'] for m in models)

        print("✅ Ollama is running!")
        print(f"\n📊 Installed models: {len(models)} ({total_size / 1e9:.1f} GB total)")

        for m in models:
            print(f"   - {m['name']}: {m['size'] / 1e9:.2f} GB")

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama")
        print("   Start it with: ollama serve")


check_ollama()