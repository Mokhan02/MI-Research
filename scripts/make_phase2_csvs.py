# scripts/make_phase2_csvs.py
"""
Convert phase2 prompt .txt files into CSVs with (prompt, target) columns.
Arithmetic prompts get auto-computed targets.
Capital prompts get auto-derived targets from a country->capital map.
Others get empty target (model top-1 used as fallback at runtime).

Usage:
  python scripts/make_phase2_csvs.py
"""
import re
import pandas as pd
from pathlib import Path

PROMPT_DIR = Path("data/prompts")
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Arithmetic ---
def parse_arith_target(prompt: str) -> str:
    """Extract answer from prompts like 'One plus one equals' or '17 + 25'."""
    # Try numeric form first: "17 + 25"
    m = re.search(r"(-?\d+)\s*([\+\-\*])\s*(-?\d+)", prompt)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        if op == "+": return str(a + b)
        elif op == "-": return str(a - b)
        elif op == "*": return str(a * b)

    # Try word form: "One plus one equals" / "Two and two make"
    word_to_num = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20,
    }
    low = prompt.lower()
    m = re.search(r"(\w+)\s+(?:plus|and)\s+(\w+)", low)
    if m:
        w1, w2 = m.group(1), m.group(2)
        if w1 in word_to_num and w2 in word_to_num:
            return str(word_to_num[w1] + word_to_num[w2])

    m = re.search(r"(\w+)\s+(?:minus|less)\s+(\w+)", low)
    if m:
        w1, w2 = m.group(1), m.group(2)
        if w1 in word_to_num and w2 in word_to_num:
            return str(word_to_num[w1] - word_to_num[w2])

    m = re.search(r"(\w+)\s+(?:times|multiplied by)\s+(\w+)", low)
    if m:
        w1, w2 = m.group(1), m.group(2)
        if w1 in word_to_num and w2 in word_to_num:
            return str(word_to_num[w1] * word_to_num[w2])

    return ""

# --- Capitals ---
CAPITAL_MAP = {
    "france": "Paris", "germany": "Berlin", "italy": "Rome",
    "spain": "Madrid", "japan": "Tokyo", "united kingdom": "London",
    "uk": "London", "china": "Beijing", "russia": "Moscow",
    "brazil": "Brasilia", "india": "New Delhi", "canada": "Ottawa",
    "australia": "Canberra", "mexico": "Mexico City",
    "south korea": "Seoul", "argentina": "Buenos Aires",
    "egypt": "Cairo", "turkey": "Ankara", "poland": "Warsaw",
    "netherlands": "Amsterdam", "sweden": "Stockholm",
    "norway": "Oslo", "denmark": "Copenhagen", "finland": "Helsinki",
    "portugal": "Lisbon", "greece": "Athens", "switzerland": "Bern",
    "austria": "Vienna", "belgium": "Brussels", "ireland": "Dublin",
    "czech republic": "Prague", "romania": "Bucharest",
    "hungary": "Budapest", "thailand": "Bangkok",
    "indonesia": "Jakarta", "vietnam": "Hanoi",
    "philippines": "Manila", "malaysia": "Kuala Lumpur",
    "singapore": "Singapore", "new zealand": "Wellington",
    "south africa": "Pretoria", "nigeria": "Abuja",
    "kenya": "Nairobi", "colombia": "Bogota", "peru": "Lima",
    "chile": "Santiago", "ukraine": "Kyiv", "israel": "Jerusalem",
    "saudi arabia": "Riyadh", "iran": "Tehran", "iraq": "Baghdad",
    "pakistan": "Islamabad", "bangladesh": "Dhaka",
}

def parse_capital_target(prompt: str) -> str:
    """Derive target from capital prompts like 'The capital of France is'."""
    low = prompt.lower()
    # "The capital of X is" / "X's capital city is"
    for country, capital in CAPITAL_MAP.items():
        if country in low:
            return capital
    # "Paris is the capital of" -> target is the country? No, target should be
    # what the model predicts next. For "Paris is the capital of" -> "France"
    # Reverse lookup
    for country, capital in CAPITAL_MAP.items():
        if capital.lower() in low:
            return country.title()
    return ""


def txt_to_csv(txt_path: Path, csv_path: Path, target_fn=None):
    """Convert txt to csv. target_fn(prompt)->str, or None for empty target."""
    lines = [l.strip() for l in txt_path.read_text().splitlines() if l.strip()]
    rows = []
    for p in lines:
        t = target_fn(p) if target_fn else ""
        rows.append({"prompt": p, "target": t})
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    n_with_target = (df["target"] != "").sum()
    print(f"Wrote {len(rows)} rows ({n_with_target} with target) -> {csv_path}")


def main():
    txt_to_csv(PROMPT_DIR / "phase2_arithmetic.txt", OUT_DIR / "arithmetic_only.csv",
               target_fn=parse_arith_target)
    txt_to_csv(PROMPT_DIR / "phase2_capitals.txt",   OUT_DIR / "capitals_only.csv",
               target_fn=parse_capital_target)
    txt_to_csv(PROMPT_DIR / "phase2_planets.txt",     OUT_DIR / "planets_only.csv",
               target_fn=None)
    txt_to_csv(PROMPT_DIR / "phase2_control.txt",     OUT_DIR / "control_only.csv",
               target_fn=None)


if __name__ == "__main__":
    main()
