# scripts/make_phase2_csvs.py
"""
Build prompt CSVs with (prompt, target) and consistent " Answer:" scaffolding.

- Adds " Answer:" to every prompt so formatting is identical across domains
  (required for neutral/task contrast and for α* logit-based design).
- Strips trailing commas from source lines.
- Only includes rows with non-empty target for planets/capitals (labeled only).
- Control/neutral: same scaffolding, target empty (structurally isomorphic).

Usage:
  python scripts/make_phase2_csvs.py

Then run domain splits so select/alpha/holdout use these labeled prompts:
  python scripts/make_domain_splits.py --data_dir data --out_dir data/prompts [--seed 0]
"""
import re
import pandas as pd
from pathlib import Path

PROMPT_DIR = Path("data/prompts")
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Scaffolding: identical across domains so last-token / last-N activations are comparable
ANSWER_SUFFIX = " Answer:"

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


# --- Planets (prompt stem -> single-token completion where unambiguous) ---
PLANET_TARGET_MAP = {
    "the largest planet in our solar system is": "Jupiter",
    "the biggest planet is": "Jupiter",
    "the fifth planet from the sun is": "Jupiter",
    "the solar system's largest planet is": "Jupiter",
    "the red spot is on": "Jupiter",
    "jupiter is the": "largest",
    "saturn is a": "planet",
    "earth is the third": "planet",
    "the moon orbits": "Earth",
    "venus is the second": "planet",
    "mercury is the first": "planet",
    "pluto is a dwarf": "planet",
    "jupiter is a": "planet",
    "neptune is a": "planet",
    "mars is the fourth": "planet",
    "neptune is the eighth": "planet",
    "earth orbits the": "Sun",
    "the sun is at the center of the": "solar",
    "the solar system includes": "planets",
    "planets orbit the": "Sun",
    "europa is a moon of": "Jupiter",
    "titan orbits": "Saturn",
    "enceladus is a moon of": "Saturn",
    "phobos and deimos orbit": "Mars",
    "iapetus is a moon of": "Saturn",
    "dione orbits": "Saturn",
    "proteus orbits": "Neptune",
    "the kuiper belt is beyond": "Neptune",
    "venus rotates": "retrograde",
    "earth is the only": "planet",
    "the sun is a": "star",
    "stars are not": "planets",
    "jupiter is gas": "giant",
    "saturn is mostly": "hydrogen",
    "mars is red": "planet",
    "earth has one": "moon",
    "the asteroid belt is between": "Mars",
    "ceres is in the": "asteroid",
    "a planet has a round": "shape",
    "orbits are": "elliptical",
    "neptune is blue": "planet",
    "uranus is an ice": "giant",
}


def parse_planet_target(prompt: str) -> str:
    """Return single-token target for planet prompt if known, else ""."""
    stem = prompt.strip().rstrip(",").lower()
    return PLANET_TARGET_MAP.get(stem, "")


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


def _norm_line(line: str) -> str:
    """Strip and remove trailing comma so 'Jupiter is the,' -> 'Jupiter is the'."""
    return line.strip().rstrip(",").strip()


def txt_to_csv(txt_path: Path, csv_path: Path, target_fn=None, labeled_only: bool = False):
    """
    Convert txt to csv with prompt = line + ANSWER_SUFFIX, target from target_fn(line).
    If labeled_only=True, only write rows where target is non-empty.
    """
    lines = [_norm_line(l) for l in txt_path.read_text().splitlines() if _norm_line(l)]
    rows = []
    for line in lines:
        prompt = line + ANSWER_SUFFIX
        t = target_fn(line) if target_fn else ""
        if labeled_only and not t:
            continue
        rows.append({"prompt": prompt, "target": t})
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    n_with = (df["target"] != "").sum() if len(df) > 0 else 0
    print(f"Wrote {len(rows)} rows ({n_with} with target) -> {csv_path}")


def main():
    txt_to_csv(PROMPT_DIR / "phase2_arithmetic.txt", OUT_DIR / "arithmetic_only.csv",
               target_fn=parse_arith_target, labeled_only=False)
    txt_to_csv(PROMPT_DIR / "phase2_capitals.txt", OUT_DIR / "capitals_only.csv",
               target_fn=parse_capital_target, labeled_only=True)
    txt_to_csv(PROMPT_DIR / "phase2_planets.txt", OUT_DIR / "planets_only.csv",
               target_fn=parse_planet_target, labeled_only=True)
    txt_to_csv(PROMPT_DIR / "phase2_control.txt", OUT_DIR / "control_only.csv",
               target_fn=None, labeled_only=False)


if __name__ == "__main__":
    main()
