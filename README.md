# SunFix: battery end-of-life prediction from field telemetry (research code)

Applies BattGP (Schaeffer et al. 2024) to the Bboxx lead-acid dataset of Aitio & Howey (2021). Data licence CC BY-NC 4.0, so research use only; the paper lists a patent application (GB 2105995.1).

**Start here:** `results/SunFix_findings_summary.pdf` (2 pages). Details in `docs/`.

| Phase | Status | Report |
|---|---|---|
| 0 Setup, BattGP and Aitio notes | done | docs/battgp_notes.md, docs/aitio_protocol.md |
| 1 Data pipeline and EDA | done | docs/eda_report.md |
| 2 Resistance model (dev subset of 60) | done | docs/phase2_report.md |
| 3 EOL prediction, all 1,027 batteries | done | docs/phase3_report.md |
| 4 Minimum-data experiments (40% subsample) | done | docs/phase4_report.md |
| 5 REPORT.md | not started | |

## Setup
Needs the raw `set_*.zip`, `meta_data.csv`, `GITT_OCV.mpt` in this folder (not in git) and `external/BattGP` cloned (commit 6d5e1db, not in git: `git clone https://github.com/JoachimSchaeffer/BattGP external/BattGP`). Python 3.12 venv with `uv pip install -r requirements.txt`. Run `make all`.
Package is `sunfix/` (not `src/`) because BattGP imports a top-level `src`.

## Key decisions and findings
- Negative current = charging. Failed batteries are truncated at the repair date (telemetry continues a median 89 days afterwards).
- BattGP is not OCV-free; we rebuild Aitio's SOC (Coulomb counting + GITT OCV curve).
- Added a per-segment random offset to the resistance model. See docs/phase2_report.md for all deviations.
- `stale_days`/`unc` features leak (data availability) and are excluded from the headline set.
- Headline: balanced accuracy 81 / 79 / 74 % at 0 / 28 / 56 days (paper 82 / 76 / 73).
- Phase 4 (40% subsample, +-4 pt intervals): no tested degradation (5 min to 1 h sampling, daily summaries, no temperature, 30-180 d history, rounding, 10-50% packet loss) measurably hurt the 28-day warning (77-81% vs 78% native); usage-only is 61-72%. Smallest passing spec: one daily record (79% subsample, 75% full data). Combinations untested.
- Generated data lives in `data/` (git-ignored, about 6 GB): parquet cache, segments, resistance trajectories, per-fold hyperparameters.
