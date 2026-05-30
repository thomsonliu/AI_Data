import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from typing import Dict, List

import pandas as pd


@dataclass
class Profile:
    rows: int
    cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_by_column: Dict[str, int]


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {path}")

    return pd.read_csv(path)


def make_profile(df: pd.DataFrame) -> Profile:
    missing = df.isna().sum().to_dict()
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}

    return Profile(
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        columns=list(df.columns),
        dtypes=dtypes,
        missing_by_column={k: int(v) for k, v in missing.items()},
    )


def profile_to_markdown(p: Profile) -> str:
    lines = []
    lines.append("# Data Profile")
    lines.append("")
    lines.append(f"- Rows: {p.rows}")
    lines.append(f"- Columns: {p.cols}")
    lines.append("")
    lines.append("## Columns")
    lines.append("")
    lines.append("| column | dtype | missing |")
    lines.append("|---|---|---:|")
    for col in p.columns:
        lines.append(f"| {col} | {p.dtypes.get(col, '')} | {p.missing_by_column.get(col, 0)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a CSV and write reproducible outputs")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output_dir", default="output", help="Directory to write outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(input_path)
    p = make_profile(df)

    (output_dir / "profile.json").write_text(json.dumps(asdict(p), indent=2, sort_keys=True))
    (output_dir / "profile.md").write_text(profile_to_markdown(p))

    print(f"Wrote: {(output_dir / 'profile.json').as_posix()}")
    print(f"Wrote: {(output_dir / 'profile.md').as_posix()}")


if __name__ == "__main__":
    main()