# Claude Code prompt: Phase 6 for SunFix (false positives, benchmark models, LFP transfer)

Paste everything below the line into Claude Code, opened in `~/Desktop/SunFix_coding`.

---

You are continuing the SunFix battery-prediction project in `~/Desktop/SunFix_coding`.

**Before anything else:**
1. Follow the session protocol in `CLAUDE.md`. Read `README.md`, the last three entries of `LOG.md` and `METHODOLOGY.md`, then give me the 5-line status summary.
2. If `CLAUDE.md` does not exist, or Phases 0–5 are not complete, stop and tell me. Phase 6 builds on:
   - the BattGP resistance model on the Bboxx data
   - the Phase 3 classifier
   - the Phase 4 minimum-data table

   The original instructions are in `claude-code-prompt-battgp-bboxx.md`; I'll paste it if needed.
3. If `papers/aitio_howey_2021_ORA_accepted_manuscript.pdf` exists, use its Supplementary Material:
   - S6 for the test set and confusion matrix
   - S8 for data-quality sensitivity

**Why Phase 6:** for SunFix, false positives are the commercial risk.
- Every false alarm sends a technician on a costly rural trip.
- Aitio & Howey's 73% balanced accuracy at 8 weeks implies a ~28.9% false-positive rate, measured on a roughly 50/50 failed/healthy sample.
- At a realistic monthly failure rate (e.g. 2%), that is about 19 wasted trips per useful one.
- Reaching 50% precision needs about 98.5% specificity.

Phase 6 therefore:
- makes the evaluation cost-aware
- benchmarks alternative models
- tests whether the method transfers to LFP, the chemistry new solar home systems ship with

The second question is whether cheap, pack-level telemetry is enough for LFP.

## Phase 6: False-positive economics, benchmark models and LFP transfer (stop after each sub-phase)

Context: Phases 0–5 reproduced Aitio & Howey with BattGP on the Bboxx lead-acid data and produced the
minimum-data table. Phase 6 makes the evaluation cost-aware, adds benchmark models and tests transfer to
LFP. Follow the session protocol in CLAUDE.md. Add a METHODOLOGY.md "Phase 6" section and a dated
Decisions entry for every choice. All public LFP datasets used here are research-only unless stated;
record each dataset's licence in README.md > Data.

### 6a. Cost-aware evaluation harness (Bboxx)
1. Create `config/costs.yaml` with PLACEHOLDER values clearly marked "hypothetical – replace with operator
   data": C_visit, C_check (remote test/SMS), C_batt_planned, C_batt_emergency, C_outage_per_day,
   D_outage_days, C_churn, technician capacity K per 1,000 systems/month.
2. Write `src/eval_ops.py`:
   - Landmark sampling over ALL system-days (realistic prevalence), not a 50/50 sample. Report the empirical
     monthly failure hazard by battery age.
   - Temporal split (train on earlier activation cohorts, test on later ones) plus a leave-one-district-out
     split if location metadata exist. Never split by row.
   - Calibration: Platt and isotonic on a held-out calibration fold; reliability diagram, Brier score, ECE.
   - Grid search over threshold τ × persistence N∈{1,3,5,7} days × confirmation {off, on with assumed
     sensitivity s_c and cost C_check} × capacity K. Output E[C] per 1,000 systems/month, false dispatches
     and missed failures per 1,000/month, precision, recall, lead-time distribution (median, p10, share ≥30 d),
     alert flips per system-month, bootstrap 95% CIs.
   - Baselines: reactive-only, age-based replacement, and Elkan threshold p* = C_FP/(C_FP+C_FN).
   - ±50% sensitivity on every cost and on the base rate; plot the break-even region.
3. Re-express the Phase 3 result in these terms. Deliver `results/ops_metrics_bboxx.csv`,
   `results/figures/cost_vs_threshold.png`, `results/figures/leadtime_hist.png`. Stop.

### 6b. Benchmark models on the same landmarks and features
1. Build `src/features.py` (no look-ahead): GP resistance mean, variance, 7/30/90-day slope, curvature,
   CUSUM statistic; partial-charge Ah between fixed voltage thresholds; end-of-discharge V at matched load;
   equivalent full cycles; time at low/high SOC and LVD trips; time above 35/40 °C; data gaps and comms
   outages; fleet-relative z-score of resistance vs a peer cohort (same install month ± 1).
2. Models: (i) the existing GP classifier; (ii) LightGBM discrete-time hazard ("fails within 30/56 d");
   (iii) Cox PH and random survival forest (scikit-survival), handling censored alive batteries properly;
   (iv) unsupervised CUSUM / Bayesian online change-point on the GP resistance trajectory plus the fleet
   z-score; (v) a 2-of-3 agreement ensemble.
3. Report balanced accuracy (for comparability with 82%/73%), C-index, time-dependent AUC, integrated
   Brier score AND the 6a operational metrics at each model's cost-optimal operating point. SHAP for
   LightGBM. Deliver `results/benchmark_table.csv`. Stop.

### 6c. Conformal gate and persistence on autocorrelated scores
1. Split-conformal (and an adaptive/online variant) calibrated on healthy system-days. Measure the
   realised false-alarm rate on later months to quantify drift and seasonality.
2. Measure the real (autocorrelated) effect of persistence N on FPR and lead time. Compare with i.i.d.
   theory (Adnan et al. 2011: n=1 → 25.3%, n=3 → 6.4%, n=5 → 1.0% at equal FAR/MAR). Stop.

### 6d. LFP transfer on the BattGP field set (Zenodo 13715694, CC BY-NC 4.0, 1.7 GB)
1. Download into `data/battgp_lfp/` (ask me first). Reproduce the paper's per-cell fault probabilities
   for 2–3 systems (e.g. 3, 6, 9) as a correctness check. Reconcile 28 vs 29 systems and the sampling
   rate from the README.
2. KEY EXPERIMENT – pack-only ablation: build pack voltage = Σ cell voltages and drop per-cell data. Fit
   the pack-level OCV-free resistance GP (the SHS-realistic setting). Question: does the pack-level signal
   detect, and how many days later, the cell faults that the per-cell weakest-link method finds? Treat the
   per-cell method as pseudo-labels. Report detection rate, delay distribution and false flags.
3. Apply Phase 4 degradations (sampling interval, no temperature, history length, rounding, packet loss)
   to the pack-only LFP model. Deliver `results/minimum_data_table_lfp.csv`.
4. Lead-acid→LFP: apply Bboxx-trained models (6b) to BattGP features after per-system normalisation;
   report rank agreement only (no labels). Stop.

### 6e. RWTH home-storage field data (Zenodo 12091223, CC BY 4.0, ~21 GB – ask before downloading)
1. Identify the "Medium_LFP" systems from Metadata_and_Code.zip; download only those.
2. Fit the OCV-free pack resistance GP; compare DCR growth with Figgener et al. (LFP +0.8–1.8 pp/yr) as an
   external validity check. Quantify temperature confounding (housing T). Stop.

### 6f. (Optional) Temperature and DoD priors from Sandia LFP cells
Verify the Battery Archive licence first. Fit resistance/capacity-fade vs temperature (15/25/35 °C) and
DoD window. Export priors for the stress-factor features. Stop.

### 6g. Write-up and operator pack
1. Update REPORT.md with the cost-based results, benchmark table and pack-only LFP finding. State plainly
   whether pack-level telemetry is sufficient for LFP.
2. Create `docs/operator_data_request.md` (telemetry incl. per-cell V if available, BMS flags, repair
   records with {battery/other/no fault} outcome, trip cost by distance band, battery price, PAYGo
   instalment and outage repayment loss, churn, fleet size by age) and `docs/shadow_mode_protocol.md`
   (2–3 months of logged-not-acted alerts before any dispatch).
3. Update README, METHODOLOGY and LOG; commit.

## Rules for Phase 6 (in addition to CLAUDE.md)
- Never modify or delete the original zips or `meta_data.csv`, or anything under `data/battgp_lfp/` or `data/rwth/` once downloaded.
- Ask before any download over 1 GB, any step over 30 minutes, or anything using more than about 16 GB of RAM. A MacBook Air has limited RAM, so stream data and process one system at a time.
- All cost values stay marked "hypothetical" until I supply operator figures. Never present hypothetical-cost results as findings.
- Licences:
  - Bboxx data (CC BY-NC) and BattGP data (CC BY-NC): research only.
  - RWTH data: CC BY 4.0.
  - Sandia, HUST and MATR: verify before use.

  Record each licence in `README.md` under Data.
- No look-ahead. Split by battery or system, and by time, never by row. Fix all seeds.
- Explain every result in plain language for an operator, an investor and a non-engineer. Lead with:
  - false dispatches per 1,000 systems per month
  - missed failures
  - median warning time

Start with the session-start summary, then 6a.
