# Dead Source Modules (Archived)

## `scoring.py`

`PlaceholderScorer` that returns deterministic hash-based proxy scores.
Only imported by Phase 1 placeholder scripts (`03_steerability.py`,
`04_offtarget.py`), which have themselves been archived.
Phase 2/3 scripts do not use any scorer — they measure on-target activation
deltas directly.

## `geometry.py`

Provided `compute_max_cosine_similarity`, `compute_neighbor_density`, and
`compute_coactivation_correlation`. Only imported by `02_presteering_metrics.py`
(archived). The geometry computation used in the active pipeline lives inline
in `scripts/phase3_predictability.py:compute_geometry_chunked()`, which
operates on NumPy arrays loaded from the SAE decoder NPZ and is vectorised
over feature chunks.
