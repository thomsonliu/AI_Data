import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError


class Extracted(BaseModel):
    person: Optional[str]
    company: Optional[str]


def parse_and_validate(text: str) -> Extracted:
    data = json.loads(text)
    return Extracted.model_validate(data)


def save_failure(raw: str):
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    path = output_dir / "raw_failure.txt"
    path.write_text(raw)

    return path


def extract_with_repair(text, call_llm, max_retries=2):
    prompt = (
        "Return ONLY valid JSON.\n"
        "Keys: person, company.\n"
        "Use null if unknown.\n\n"
        f"{text}"
    )

    last_error = None

    for _ in range(max_retries + 1):
        raw = call_llm(prompt)

        try:
            return parse_and_validate(raw)

        except (json.JSONDecodeError, ValidationError) as e:
            save_failure(raw)

            last_error = str(e)

            prompt = (
                "Your previous output was invalid.\n"
                "Fix it and return ONLY valid JSON.\n\n"
                f"Invalid output:\n{raw}\n\n"
                f"Error:\n{last_error}"
            )

    raise ValueError(
        f"Failed after retries. Last error: {last_error}"
    )
