"""Check overlap between pilot uncensored features and full run selected features."""
import json

pilot_uncensored = {1645, 14481, 1216, 3917, 9348, 10461, 10648, 10936, 13263, 15959}

with open("outputs/phase2_salad_full/selected_features.json") as f:
    full_features = set(json.load(f))

try:
    with open("outputs/phase2_select_salad/selected_features_salad.json") as f:
        pilot_features = set(json.load(f))
except FileNotFoundError:
    pilot_features = None

print(f"Full run features: {len(full_features)}")
overlap_uncensored = pilot_uncensored & full_features
print(f"Pilot uncensored in full run: {len(overlap_uncensored)}/{len(pilot_uncensored)}")
print(f"  In both: {sorted(overlap_uncensored)}")
print(f"  Missing: {sorted(pilot_uncensored - full_features)}")

if pilot_features:
    overlap_all = pilot_features & full_features
    print(f"Total pilot features in full run: {len(overlap_all)}/{len(pilot_features)}")
