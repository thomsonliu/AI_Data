import json
import re
import requests
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError


# -------------------------
# 1. Schema
# -------------------------
class Extracted(BaseModel):
    person: Optional[str]
    company: Optional[str]


# -------------------------
# 2. Ollama call
# -------------------------
def call_llm(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False
        }
    )
    response.raise_for_status()
    return response.json()["response"]


# -------------------------
# 3. Safe JSON extraction (important for Ollama)
# -------------------------
def safe_json_extract(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON found in model output")
    return json.loads(match.group(0))


# -------------------------
# 4. Parse + validate
# -------------------------
def parse_and_validate(text: str) -> Extracted:
    data = safe_json_extract(text)
    return Extracted.model_validate(data)


# -------------------------
# 5. Save failures
# -------------------------
def save_failure(raw: str):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    path = output_dir / "raw_failure.txt"
    path.write_text(raw)

    return path


# -------------------------
# 6. Main extraction with retry
# -------------------------
def extract_with_repair(text, call_llm, max_retries=2):
    prompt = (
        "You are a strict JSON extraction engine.\n"
        "Return ONLY valid JSON. No explanation.\n"
        "Schema:\n"
        '{"person": string or null, "company": string or null}\n\n'
        f"Input:\n{text}\n"
    )

    last_error = None

    for _ in range(max_retries + 1):
        raw = call_llm(prompt)

        try:
            return parse_and_validate(raw)

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            save_failure(raw)
            last_error = str(e)

            prompt = (
                "Your previous output was invalid.\n"
                "Fix it and return ONLY valid JSON.\n"
                "No extra text.\n\n"
                f"Invalid output:\n{raw}\n\n"
                f"Error:\n{last_error}\n"
            )

    raise ValueError(f"Failed after retries. Last error: {last_error}")


# -------------------------
# 7. CLI usage (IMPORTANT PART)
# -------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input text or JSON file")
    parser.add_argument("--json", action="store_true", help="Treat input as JSON file")
    args = parser.parse_args()

    path = Path(args.input)

    # Read input
    if path.exists():
        raw_input = path.read_text()
    else:
        raw_input = args.input

    # If JSON mode
    if args.json:
        data = json.loads(raw_input)
        raw_input = json.dumps(data)

    result = extract_with_repair(raw_input, call_llm)

    print("\n✅ RESULT:")
    print(result.model_dump())
