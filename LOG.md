# Lab notebook (append-only)

Sessions 1-4 were not logged at the time. Entries below were reconstructed on 2026-11-03 from git history and are marked (reconstructed).

## 2026-11-02 19:23 – Session 1 (Phase 0) (reconstructed)
**Goal:** setup, BattGP and Aitio notes. **Done:** repo, venv, `docs/battgp_notes.md`, `docs/aitio_protocol.md`. **Commit:** 93cf3f8, 9b379b0

## 2026-11-02 19:33 – Session 2 (Phase 1) (reconstructed)
**Goal:** data pipeline. **Done:** loader, cleaning, Aitio segment extraction, EDA (`docs/eda_report.md`). **Commit:** bbea8e2, 9e2b6d8

## 2026-11-02 19:46 – Session 3 (Phase 2) (reconstructed)
**Goal:** BattGP resistance model on 60 dev batteries. **Results:** `docs/phase2_report.md`; ocv variant predicts voltage 26% better than nocv; 0.76 s/battery. **Decisions:** per-segment offset. **Commit:** aad7bee

## 2026-11-02 20:13 – Session 4 (Phase 3) (reconstructed)
**Goal:** EOL prediction on all 1,027 batteries. **Results:** balanced accuracy 81 / 79 / 79 / 76 / 74% at h = 0/14/28/42/56 (paper 82/80/76/76/73); usage-only about 70.5%. **Problems:** `stale_days` leak found and removed. **Commit:** 531d19e

## 2026-11-03 – Session 5 (Phase 4 start)
**Goal:** reconcile documents (CLAUDE.md, LOG.md, METHODOLOGY.md were never created), publish repo to GitHub, design and start Phase 4.
**Done:** created the three documents; see below for continuation.
**Problems / surprises:** the three files were missing although expected; reconstructed. The original brief was missing from the folder until the user added it.
**Done:** Phase 4 harness (`sunfix/phase4.py`, `phase4_report.py`), 15 scenarios on a 40% stratified subsample; report `docs/phase4_report.md`; repo pushed to GitHub (private, matteokuhn-gif/SunFix).
**Results:** no scenario measurably below native (full model 77-81% at 28 d vs 78% native, +-4 pt); usage-only 61-72%; one daily record gives 79% (subsample) / 75% (full data, 56 d: 70.8%). `results/minimum_data_table_sub40.csv`, `results/figures/phase4_minimum_data_sub40.png`.
**Problems / surprises:** joblib workers did not see runtime-registered feature sets (moved to `features.SETS`); empty 30-day windows crashed stress features (guard added). Run crashed once and was restarted; finished scenarios were kept.
**Decisions:** subsample, daily proxy, gap scaling (see METHODOLOGY.md).
**Next:** Phase 5 REPORT.md; optionally a combined cheap-device scenario and a full-data run of the key scenarios; Phase 6.
**Commit:** see git log

## 2026-11-03 – Session 6 (Phase 5, Phase 6a)
**Goal:** write REPORT.md; cost-aware evaluation harness (6a); stop after 6a per PHASE6_PROMPT.md.
**Done:** REPORT.md; `config/costs.yaml` (hypothetical); `sunfix/landmarks.py`, `eval_ops.py`, `run_6a.py`; `docs/phase6a_report.md`; figures cost_vs_threshold, leadtime_hist, calibration_bboxx, breakeven_bboxx.
**Results:** 434k landmarks; hazard 0 below 360 d, 4.9-14% per month above (as sampled). Phase-3-style operating point: 67 false vs 17 true dispatches per 1,000 per month (precision 21%), cost above reactive under hypothetical costs. Best cost-optimal policy about 9% below reactive (hypothetical). Calibration test ECE 0.006.
**Problems / surprises:** age-based baseline equals reactive (placeholder planned/emergency costs too close; test cohort younger). Sample is 48% failed, so as-sampled hazard overstates a real fleet. No district metadata: leave-one-district-out skipped.
**Decisions:** see METHODOLOGY.md Phase 6 and Decisions.
**Next:** 6b benchmark models (features.py with no look-ahead, LightGBM hazard, Cox/RSF, CUSUM, ensemble).
**Commit:** see git log

## 2026-11-03 – Session 6 continued (Phase 6b)
**Goal:** benchmark models on the 6a landmarks.
**Done:** `sunfix/features6.py`, `bench.py`, `bench_ibs.py`; `results/benchmark_table.csv`; SHAP; `docs/phase6b_report.md`; small derived data tracked in `derived/`.
**Results:** at-risk balanced accuracy 75-78% for all supervised models; usage-only 76%, age only 65%; CUSUM+z 58%; ensemble no gain; IBS Cox 0.0254, RSF 0.0223. Cost-optimal operating points indistinguishable (hypothetical costs).
**Problems / surprises:** unrestricted balanced accuracy 88-91% is an age artefact (caught from SHAP). Installing scikit-survival upgraded numpy to 2.5 and broke the resistance model; restored numpy 1.26.4 and re-verified Phase 3 equivalence (diff 1e-10). First bench run swallowed metric errors (np.trapezoid, test-event filter); fixed. Detached jobs do not notify on completion; use a watcher.
**Decisions:** see METHODOLOGY.md.
**Next:** 6c conformal gate and persistence on autocorrelated scores; 6d needs approval for a 1.7 GB download.
**Commit:** see git log

## 2026-11-03 – Session 6 continued (Phase 6c)
**Goal:** conformal gate and persistence on autocorrelated scores.
**Done:** `sunfix/conformal.py`; `docs/phase6c_report.md`; `results/conformal_far_by_month.csv`, `persistence_bboxx.csv`; two figures.
**Results:** static conformal realised false-alarm rate 3.7% at alpha 5% overall but 2.5% / 1.8% (May, Jun) vs 5.0% / 6.2% (Jul, Aug); age-conditional best; sliding window 14% at 5% target. Persistence N=3/5/7 cuts FPR by 12/21/29% (i.i.d. would predict 96%+); at equal budget no detection gain, lead time 30 -> 26 d.
**Problems / surprises:** only five months of at-risk test data, so seasonality is confounded with age. Cited Adnan et al. numbers not reproducible from p^N; definition unverified.
**Next:** 6d: LFP transfer on the BattGP field set. Needs approval for a 1.7 GB download (Zenodo 13715694, CC BY-NC 4.0).
**Commit:** see git log

## 2026-11-03 – Session 6 continued (Phase 6d.1)
**Goal:** download BattGP LFP data (approved by the user: 1.7 GB) and reproduce per-cell fault probabilities for systems 3, 6, 9.
**Done:** download + MD5 check, data made read-only; `sunfix/lfp_reproduce.py`; ran systems 3, 9, 6 (13 s, 36 s, 2.3 min; peak 6.5 GB); `docs/phase6d_report.md`.
**Results:** relative band probabilities flag cell 8 (system 6) and cell 2 (system 9); system 3 no clear fault. 28 vs 29: zip has 28; BattGP runner loops 1-29. Sampling median 5 s.
**Problems / surprises:** paper not readable by the fetch agent (HTTP 403), no local copy, so comparison with the paper's numbers is NOT done. Absolute-threshold probability saturates at 1.0 for all cells.
**Decisions:** none new methodologically; BattGP run unchanged with its shipped settings.
**Next:** obtain the paper (user to supply PDF); then 6d.2 pack-only ablation.
**Commit:** see git log

## 2026-11-03 – Session 6 continued (Phase 6d.1 completed)
**Done:** read the paper (arXiv 2406.19015v3) via a fetch agent; ran BattGP on systems 3, 9, 6 under three settings; compared with the paper's statements.
**Results:** cell identification and onset timing match the paper under the paper's settings (cell 7 / sys 3; cell 8 / sys 6 crosses 0.5 on day 1020 vs "shortly before 1000"; cell 2 / sys 9 on day 241 vs 240). Under BattGP default settings system 6 cell 8 is flagged from the start (reference operating point matters). Paper: 29 systems/131M rows vs data 28/133M.
**Problems / surprises:** absolute 2 mOhm threshold is not in the paper and saturates (cell R0 about 8-9 mOhm). Paper figure captions and text disagree on system numbers.
**Decisions:** use paper settings with mean reference operating point and a 30-day burn-in for per-cell pseudo-labels in 6d.2.
**Commit:** see git log

## 2026-11-03 – Session 6 continued (Phase 6d.2 to 6d.4, full report)
**Goal:** pack-only ablation on LFP, degraded pack-only data, lead-acid to LFP rank agreement, full PDF report.
**Done:** per-cell BattGP on all 28 systems (21 modelled; 22-28 fail the minimum-data filter); drift-based pseudo-labels; pack-only pipeline (`lfp_pack.py`, `lfp_pack_run.py`, `lfp_pack_degrade.py`, `lfp_sections.py`, `lfp_transfer.py`); `docs/FULL_REPORT.md` and PDFs (`python -m sunfix.md2pdf`).
**Results:** pack-only detection 0-4 of 10 faults, 0-6 of 8 controls falsely flagged, AUC 0.17-0.92 across fit sets: inconclusive, leaning negative. Baselinable systems fall from 17 to 10 / 7 / 8 at 5 min / 15 min / 1 h sampling. Transfer: Spearman -0.34 (CI -0.74 to 0.33), AUC 0.50.
**Problems / surprises:** paper-literal weakest-link flags 19/21 systems from day 30 (not usable as a label); pack pipeline bugs (1000x time units, quadratic rest lookup, lead-acid hyperparameter bound pinning tau); method revised several times on the same 19 systems, so the pack-only result is exploratory; first degradation table had a fixed 5-min rest lookback that made coverage zero at slow sampling (scaled and re-run).
**Decisions:** see METHODOLOGY.md.
**Next:** 6g (final write-up with Phase 6, operator data request, shadow-mode protocol); decide on 6e/6f; see section 7 of the full report.
**Commit:** see git log
