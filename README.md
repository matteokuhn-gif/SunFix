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
| 5 REPORT.md | done | REPORT.md |
| 6a Cost-aware evaluation (hypothetical costs) | done | docs/phase6a_report.md |
| 6b Benchmark models | done | docs/phase6b_report.md |
| 6c Conformal gate, persistence | done | docs/phase6c_report.md |
| 6d LFP transfer | in progress: data downloaded, BattGP reproduction runs, paper comparison pending | docs/phase6d_report.md |
| 6e-6g RWTH, priors, write-up | not started | PHASE6_PROMPT.md |

## Setup
Needs the raw `set_*.zip`, `meta_data.csv`, `GITT_OCV.mpt` in this folder (not in git) and `external/BattGP` cloned (commit 6d5e1db, not in git: `git clone https://github.com/JoachimSchaeffer/BattGP external/BattGP`). Python 3.12 venv with `uv pip install -r requirements.txt`. Run `make all`.
Package is `sunfix/` (not `src/`) because BattGP imports a top-level `src`.

## Data licences
- Bboxx / Aitio & Howey 2021: CC BY-NC 4.0, research only (patent application GB 2105995.1). Raw data is not in git.
- BattGP LFP field data (Zenodo 13715694): CC BY-NC 4.0, research only. Downloaded 2026-11-03 to data/battgp_lfp/ (read-only, MD5 verified, not in git).
- RWTH home storage (Zenodo 12091223): CC BY 4.0 (not yet downloaded). Sandia / HUST / MATR: verify before use.
- Operating costs in `config/costs.yaml` are hypothetical placeholders; results based on them are not findings.

## Key decisions and findings
- Negative current = charging. Failed batteries are truncated at the repair date (telemetry continues a median 89 days afterwards).
- BattGP is not OCV-free; we rebuild Aitio's SOC (Coulomb counting + GITT OCV curve).
- Added a per-segment random offset to the resistance model. See docs/phase2_report.md for all deviations.
- `stale_days`/`unc` features leak (data availability) and are excluded from the headline set.
- Headline: balanced accuracy 81 / 79 / 74 % at 0 / 28 / 56 days (paper 82 / 76 / 73).
- Phase 4 (40% subsample, +-4 pt intervals): no tested degradation (5 min to 1 h sampling, daily summaries, no temperature, 30-180 d history, rounding, 10-50% packet loss) measurably hurt the 28-day warning (77-81% vs 78% native); usage-only is 61-72%. Smallest passing spec: one daily record (79% subsample, 75% full data). Combinations untested.
- Phase 6a (hypothetical costs): at the as-sampled failure rate, a Phase-3-style balanced operating point gives about 67 false dispatches per 17 true per 1,000 systems per month (precision 21%); acting on it costs more than reactive-only. Details: docs/phase6a_report.md.
- Phase 6b: at-risk (age >= 360 d) balanced accuracy 75-78% for LightGBM, Cox, survival forest and GP classifier alike; usage-only 76%, age only 65%, so resistance adds about 1-2 points (noise). CUSUM + fleet z alone 58%. Overall-landmark accuracy (88-91%) is inflated by age. Details: docs/phase6b_report.md.
- Phase 6c: a static conformal gate is conservative overall but its monthly false-alarm rate was 2.5-1.8% in May-June and 5.0-6.2% in July-August (target 5%); age-conditional calibration tracks best; sliding window fails. Requiring N consecutive days cuts false positives by only 12-29% (N=3-7) versus 75-96% under i.i.d. theory, and does not improve detection at equal false-alarm budget. Details: docs/phase6c_report.md.
- Small derived data is tracked in `derived/` (see derived/README.md).
- Generated data lives in `data/` (git-ignored, about 6 GB): parquet cache, segments, resistance trajectories, per-fold hyperparameters.
