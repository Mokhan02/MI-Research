"""Check overlap between pilot uncensored features and full run features."""
import json
import pandas as pd

pilot_uncensored = {1645, 14481, 1216, 3917, 9348, 10461, 10648, 10936, 13263, 15959}

# Get actual features from feature_summary.csv (ground truth of what was run)
summary = pd.read_csv("outputs/phase2_salad_full/feature_summary.csv")
full_features = set(summary["feature_id"].astype(int).tolist())

print(f"Full run features (from feature_summary): {len(full_features)}")

overlap = pilot_uncensored & full_features
missing = pilot_uncensored - full_features
print(f"Pilot uncensored in full run: {len(overlap)}/{len(pilot_uncensored)}")
print(f"  Present: {sorted(overlap)}")
print(f"  Missing: {sorted(missing)}")

# For features that ARE present, show their full-run results
if overlap:
    print("\nFull-run results for pilot uncensored features:")
    subset = summary[summary["feature_id"].isin(pilot_uncensored)].sort_values("feature_id")
    print(subset.to_string(index=False))
