# scripts/make_phase2_csvs.py
"""
Convert phase2 prompt .txt files into CSVs with (prompt, target) columns.
Arithmetic prompts get auto-computed targets; others get empty target.

Usage:
  python scripts/make_phase2_csvs.py
"""
import re
import pandas as pd
from pathlib import Path

PROMPT_DIR = Path("data/prompts")
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_arith_target(prompt: str) -> str:
    """Extract answer from arithmetic prompts like 'What is 17 + 25? ...'"""
    m = re.search(r"(-?\d+)\s*([\+\-\*])\s*(-?\d+)", prompt)
    if not m:
        return ""
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op == "+":
        ans = a + b
    elif op == "-":
        ans = a - b
    elif op == "*":
        ans = a * b
    else:
        return ""
    return str(ans)


def txt_to_csv(txt_path: Path, csv_path: Path, with_target: bool):
    lines = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    rows = []
    for p in lines:
        if with_target:
            rows.append({"prompt": p, "target": parse_arith_target(p)})
        else:
            rows.append({"prompt": p, "target": ""})
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {len(rows)} rows -> {csv_path}")


def main():
    txt_to_csv(PROMPT_DIR / "phase2_arithmetic.txt", OUT_DIR / "arithmetic_only.csv", with_target=True)
    txt_to_csv(PROMPT_DIR / "phase2_capitals.txt",   OUT_DIR / "capitals_only.csv",   with_target=False)
    txt_to_csv(PROMPT_DIR / "phase2_planets.txt",     OUT_DIR / "planets_only.csv",    with_target=False)
    txt_to_csv(PROMPT_DIR / "phase2_control.txt",     OUT_DIR / "control_only.csv",    with_target=False)


if __name__ == "__main__":
    main()
