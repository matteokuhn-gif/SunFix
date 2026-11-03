# Derived data (small, tracked)
Features, fitted hyperparameters and experiment outputs derived from the Bboxx dataset (Aitio & Howey 2021, CC BY-NC 4.0, research use only). Contains no raw telemetry. Private repo; do not redistribute publicly without checking the data licence and the patent note.
- `landmarks.parquet`: one row per battery-day (age >= 120 d): features, labels, ages (Phase 6a). Rebuild: `python -m sunfix.landmarks`.
- `features_*.parquet`, `hyper_*.json`: Phase 3 features and per-fold resistance-GP hyperparameters.
- `phase4_sub40/`, `phase4_full/`: Phase 4 scenario results.
Raw zips (2.8 GB) and per-battery caches (3.8 GB) are NOT tracked; get the raw data from the original dataset release and rebuild with `make all`.
