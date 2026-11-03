# SunFix: predicting solar-home-system battery failure from telemetry
## Full project report (Phases 0 to 6d) and recommended way forward

Prepared 3 November 2026. Code and data notes: github.com/matteokuhn-gif/SunFix (private). All results are research findings on one lead-acid data set (Bboxx, Aitio & Howey 2021, CC BY-NC 4.0) and one LFP data set (BattGP / Zenodo 13715694, CC BY-NC 4.0). Costs used anywhere in this report are **hypothetical placeholders**, not findings.

---

## 1. Executive summary (plain language)

**What SunFix is trying to answer.** Can a solar-home-system maker or operator predict that a battery will fail weeks ahead, using only the telemetry the system already sends, and how little data is enough? For an operator the commercial risk is false alarms: every needless technician trip costs money.

**What we found, in ten points**

1. **We reproduce the published result.** On 1,027 lead-acid batteries our model predicts failure with 81% / 79% / 74% balanced accuracy at 0 / 4 / 8 weeks before failure (the paper: 82 / 76 / 73%).
2. **Most of the signal is how the battery is used, not its internal resistance.** Usage statistics alone reach about 70%; adding the resistance model adds roughly 1 to 8 points depending on how it is measured, and only 1 to 2 points in our most demanding test. The resistance model is real but a modest extra.
3. **Thin data is enough for lead-acid.** Cutting sampling from one reading per minute to one per hour, dropping temperature, rounding voltage and current, losing up to half of the packets, or keeping a single daily summary record did not measurably hurt a 4-week warning (all within about 4 points on a 40% subsample). The smallest specification tested that still works is **one daily summary record per system** (79% on the subsample; 75% on all batteries).
4. **False alarms are the real problem.** The operating point that gives the headline accuracy, re-expressed at the failure rate in this data, sends about **4 wasted trips for every useful one** (67 false vs 17 true dispatches per 1,000 systems per month). At a real fleet's lower failure rate it is worse. Acting on that operating point costs more than doing nothing under our placeholder costs.
5. **Requiring several alert days in a row does not fix it.** Because scores stay high for days, N consecutive days cuts false positives by only 12 to 29% (theory for independent days says 75 to 96%) and does not improve detection at an equal false-alarm budget. A higher threshold or an independent confirmation step (remote check) works; waiting longer does not.
6. **Fancier models do not beat simple ones.** LightGBM, Cox, a survival forest and the GP classifier all give 75 to 78% in the period when batteries can fail; the unsupervised change detector alone gives 58%; combining models does not help.
7. **The economics depend on one number we do not have:** the gap between a planned and an emergency battery replacement, plus trip cost and failure base rate. With our placeholders, the best policy is about 9% cheaper than doing nothing; with a smaller gap it saves nothing.
8. **For LFP (what new solar home systems ship with) we could not show that cheap pack-level telemetry detects single-cell faults.** BattGP reproduces on its own data; but with only pack voltage, current and temperature, a rest-voltage-based resistance estimate gave unstable results (detecting 0 to 4 of 10 faulty systems and falsely flagging 0 to 6 of 8 healthy ones, depending on a free choice) and loses about half its systems at 5-minute sampling. The honest reading: **not demonstrated, leaning negative**; the sample is small (10 vs 8 systems) and the method was revised while looking at the data.
9. **A lead-acid model does not carry over to LFP.** Its risk ranking has no agreement with the per-cell method on LFP (Spearman -0.34, interval -0.74 to +0.33).
10. **Implication for an open telemetry specification:** for lead-acid, a daily summary appears sufficient for usage-based warnings; for LFP keep per-cell voltage (or at least per-cell min/max summaries) in the specification until a pack-only method is proven on a larger labelled data set.

**What to do next, in one line:** get real operator data (repair outcomes, costs, trips) to replace the placeholders, run a 2 to 3 month shadow-mode trial before any dispatch, and test cheap per-cell summaries for LFP on a larger labelled set. Details in section 7.

---

## 2. Questions and data

**Questions.** (1) Reproduce Aitio & Howey's prediction of battery end of life using BattGP's Gaussian-process resistance model: balanced accuracy at failure, 4 weeks and 8 weeks before (paper: 82% at failure and 73% at 8 weeks). (2) The minimum telemetry a manufacturer must ship for a useful 30-day warning. Phase 6 added: make the evaluation cost-aware (false positives), benchmark other models, and test whether the method transfers to LFP.

**Lead-acid data (Bboxx).** 1,027 batteries (12 V valve-regulated lead-acid, 20 Ah); 491 entered repair (the failure label), 536 censored (alive); about 620 million rows of time, current, voltage and temperature at a median 60 s interval. The sample is about 48% failed batteries by design, so the failure rate in it overstates a real fleet. Failed batteries' telemetry continues a median 89 days after the repair date, so series are cut at the repair date. No location information exists.

**LFP data (BattGP, Zenodo 13715694).** 28 systems (24 V, about 160 Ah, 8 cells in series, one current sensor, nine voltage sensors, four temperature sensors), about 133 million rows, median 5 s sampling, all returned to the manufacturer for unsatisfactory behaviour (biased sample). 21 systems pass BattGP's minimum-data filter; the paper also models 21 (it states 29 systems and 131 million rows; the released data has 28 systems and 133 million rows).

---

## 3. What was done, phase by phase

### Phase 0 to 2: setup, data pipeline, resistance model
- Loader streams the zips without extracting; cleaning drops impossible voltages (outside 9 to 16 V), duplicate timestamps and post-repair data; charging sessions are selected as in Aitio & Howey (at least 6,000 s, rest voltage 11.5 to 12.9 V, maximum above 14 V, no gap over 610 s).
- BattGP is not OCV-free in practice (it needs an SOC signal and an OCV curve). We rebuilt state of charge by Coulomb counting with the lab OCV curve, and used BattGP's recursive spatiotemporal GP as the resistance model, plus one extension (a per-session random offset, which lifted the fit by about 1,700 log-likelihood units).
- The OCV variant predicted next-session voltage 26% better than an OCV-free variant. Runtime 0.76 s per battery.

### Phase 3: reproduction (all 1,027 batteries, battery-level 5-fold cross-validation)
| Days before failure | 0 | 14 | 28 | 42 | 56 |
|---|---|---|---|---|---|
| This work (resistance + usage, GP classifier) | 81.3% | 79.1% | 78.6% | 76.2% | 74.2% |
| Paper | 82 | 80 | 76 | 76 | 73 |
| Usage statistics only | 71.6 | 70.5 | 70.7 | 70.5 | 70.5 |
| Resistance only | 73.8 | 70.4 | 69.9 | 67.6 | 64.3 |
| Battery age only | 61.8 | | | | |
Interval about +-2.5 points. Calibration is good (ECE 0.02 to 0.03). **A leak was found and removed:** a "days since last charging session" feature gave 87% at failure alone, because failed batteries' series end at the repair date while healthy ones end at the last telemetry; it and the posterior-uncertainty column are excluded.

![Phase 3 summary](FIG/phase3_summary.png)

### Phase 4: minimum-data experiments (40% subsample, intervals about +-4 points)
Four-week-ahead balanced accuracy of the full model (usage-only in brackets):
| Telemetry | 28 days |
|---|---|
| Native 1 reading/min | 78% (71) |
| 5 / 15 / 60 min | 79 / 79 / 78% (70 / 70 / 69) |
| One daily record (min/max/mean V, I, T; Wh in/out) | 79% (70); 75% on all batteries |
| No temperature | 81% (71) |
| History 30 / 90 / 180 days | 74 / 78 / 78% (61 / 63 / 66) |
| Voltage rounded 0.1 V / current 0.1 A | 77 / 78% |
| 10 / 30 / 50% of rows lost | 77 / 78 / 77% |
No tested degradation produced a loss larger than the uncertainty; combinations of degradations were not tested. At 56 days the benefit over usage statistics is not established.

![Phase 4 minimum data](FIG/phase4_minimum_data_sub40.png)

### Phase 5: written report
`REPORT.md` (and its PDF) covers Phases 0 to 4.

### Phase 6a: false-positive economics (hypothetical costs)
Landmarks = every system-day at age 120 days or more (434,045), label "fails within 30 days", split by activation date (train / calibration / test), probabilities calibrated (test Brier 0.021, ECE 0.006). No battery failed before about 360 days; monthly failure hazard (as sampled) is 4.9% at 360 to 419 days rising to 14% at 660 days and over.
Per 1,000 systems per month, test cohort, as-sampled base rate:
| Policy | False dispatches | True | Missed | Precision | Median warning |
|---|---|---|---|---|---|
| Phase-3-style (balanced-accuracy threshold) | 67 | 17 | 12 | 21% | 33 d |
| Elkan cost threshold | 2 | 7 | 23 | 75% | 23 d |
| Cost-optimal with remote confirmation | 2 | 16 | 14 | 88% | 28 d |
| Reactive only | 0 | 0 | 29 | n/a | n/a |
Under the placeholder costs: reactive 3,832 per 1,000 per month; Phase-3-style 5,998 (56% worse than doing nothing); best policy with confirmation about 3,480 (about 9% better). The saving vanishes if planned replacement costs 50% more or emergency replacement is 50% cheaper: **the planned-versus-emergency replacement gap decides whether prediction pays.** Break-even surfaces in the figure below.

![Cost versus threshold](FIG/cost_vs_threshold.png)

![Break-even region](FIG/breakeven_bboxx.png)

### Phase 6b: benchmark models (same landmarks, same split)
At-risk period only (age 360 days or more), balanced accuracy for the 30-day label:
| Model | Balanced accuracy | AUC | C-index |
|---|---|---|---|
| LightGBM (resistance + usage) | 77.6% | 0.864 | 0.837 |
| LightGBM usage only | 76.2% | 0.857 | 0.840 |
| LightGBM age only | 64.9% | 0.687 | 0.696 |
| GP classifier (sampled) | 75.2% | 0.852 | 0.748 |
| Cox PH | 76.9% | 0.864 | 0.881 |
| Random survival forest | 76.6% | 0.851 | 0.874 |
| CUSUM + fleet z-score (no training) | 58.0% | 0.570 | 0.499 |
| 2-of-3 ensemble | 75.9% | 0.847 | 0.865 |
Over all landmarks the same models score 88 to 91% — an artefact: young systems are trivially healthy (SHAP: age 2.33 log-odds, next feature 0.30). Resistance adds about 1.4 points over usage only (within noise, about 100 failed test batteries). Cost-optimal operating points of the supervised models are indistinguishable.

![SHAP](FIG/shap_lightgbm_bboxx.png)

### Phase 6c: conformal gate and persistence
- A split-conformal gate calibrated on healthy system-days is conservative overall (3.7% realised at a 5% target) but its monthly false-alarm rate was 2.5%, 1.8%, 5.0%, 6.2% from May to August: older systems score higher, and seasonality cannot be separated from age in five months of data. Calibrating by age band tracked the target best; a sliding window failed (14% at a 5% target).
- Persistence (N consecutive days) cut the false-positive rate from 19.6% to only 17.2%, 15.4%, 13.9% at N = 3, 5, 7; i.i.d. theory predicts 0.7% and 0.03%. At an equal 5% false-alarm budget detection stayed at 66 to 69% and the median warning fell from 30 to 26 days.

![Conformal gate by month](FIG/conformal_far_by_month.png)

![Persistence effect](FIG/persistence_effect.png)

### Phase 6d: LFP transfer
- **6d.1 correctness check (passed with caveats).** BattGP runs on the Zenodo data and, with the paper's own settings, reproduces the paper's cell identification and timing: system 3 cell 7 (largest resistance rise), system 6 cell 8 crossing probability 0.5 on day 1020 (paper: "shortly before 1000 days"), system 9 cell 2 on day 241 (paper: 240). With BattGP's default fixed reference operating point, system 6 cell 8 is flagged from the start, so the reference choice matters. The repo's absolute 2 mOhm threshold saturates at 1.0 and is not in the paper. Paper read: arXiv v3 preprint.
- **6d.2 pack-only test (key experiment).** Pack voltage = sum of cell voltages; no cell data, no SOC, no OCV; rest-voltage-referenced resistance GP. Against pseudo-labels from the per-cell method (10 fault systems, 8 controls): detection 0 to 4 of 10 and false flags 0 to 6 of 8 depending on the hyperparameter fit set; AUC 0.17 to 0.92. **Inconclusive, leaning negative.** The fitted per-run reference error (18 to 55 mOhm) exceeds the pack resistance.
- **6d.3 degraded pack-only data.** Systems with enough pack observations for a baseline (of 18): 17 at native and 1-minute sampling, 10 at 5 minutes, 7 at 15 minutes, 8 at 1 hour (median observations 1,960, 996, 250, 139, 21); unaffected by history limits, rounding to 0.1 V or 1 A, or packet loss up to 50%. Detection numbers are unreliable.
- **6d.4 transfer.** The relative-resistance Bboxx model (own AUC only 0.66) ranks LFP systems with Spearman -0.34 (interval -0.74 to +0.33, p = 0.31, 11 systems) against the per-cell drift: no agreement.

![LFP pack-only](FIG/lfp_pack_only.png)

---

## 4. How far each claim can be trusted

| Claim | Confidence | Why |
|---|---|---|
| Reproduction of 81 / 79 / 74% on lead-acid | High | Full data, battery-level folds, a leak found and removed; matches the paper within noise |
| Usage statistics carry most of the signal; resistance adds little | High | Seen in Phase 3 and again in 6b under a harder test |
| Thin lead-acid telemetry is enough for a 4-week warning | Medium | 40% subsample, +-4 points, single-factor tests only, degradations emulated from 1-minute data |
| About 4 wasted trips per useful one at balanced settings; persistence does little | Medium-high | Realistic-prevalence landmarks, temporal split; but as-sampled base rate and 93 failed test batteries |
| Economics (saving of about 9%; break-even region) | Low | All costs are placeholders; the shape only |
| Fancier models do not beat simple ones | Medium | About 100 test failures; differences under 2 points not detectable |
| BattGP reproduces on LFP | Medium-high | Matches the paper's cells and timing under its settings; paper read as a preprint |
| Pack-only LFP cannot be shown to work | Medium (as "not shown") | Small sample, labels invented by us, method revised several times on the same data. It does not prove pack-level detection is impossible |
| A lead-acid model does not transfer to LFP | Medium | 11 systems, weak source model |

## 5. Problems found and corrected along the way (for transparency)
- Documents `CLAUDE.md`, `LOG.md`, `METHODOLOGY.md` did not exist and were reconstructed from the git history.
- A data-availability leak in the features (87% instead of 81% at failure) was found and removed.
- The unrestricted balanced accuracy in 6b (88 to 91%) was recognised as an age artefact and replaced by the at-risk evaluation.
- Installing one package upgraded numpy and silently broke the resistance model; numpy was pinned back and equivalence with earlier results re-verified (1e-10).
- The LFP pack pipeline had a 1000x time-unit bug, a quadratic loop, a hyperparameter bound set for lead-acid units, and an alarm rule that falsely flagged 6 of 8 controls; all were fixed, and each fix changed results, which is why the pack-only result is called exploratory.
- Phase 4 was run on a 40% subsample to save time; the benchmark (6b) and operating results use temporal, not random, splits.
- A paper mismatch (29 systems in the paper vs 28 released) was traced; the paper's own figure captions and text disagree about system numbers.

---

## 6. What is not done
- 6e (RWTH home-storage data, about 21 GB, external validity for LFP resistance growth): waiting for approval; arguably low value for the core question.
- 6f (Sandia temperature and depth-of-discharge priors): optional; licence must be verified first.
- 6g (final write-up with the Phase 6 results folded into `REPORT.md`, `docs/operator_data_request.md`, `docs/shadow_mode_protocol.md`).
- Not implemented: Bayesian online change-point detection; leave-one-district-out (no location data); a held-out, pre-specified evaluation of the pack-only LFP method; any real-cost evaluation.

---

## 7. How we should continue

### Priority A: now (days)
1. **Get real operator numbers** — the biggest lever. Send the operator a short request (to be written as `docs/operator_data_request.md`): repair records with an outcome of battery fault / other fault / no fault; trip cost by distance band; battery price, and the actual cost of a planned versus an emergency replacement; PAYGo instalment and repayment lost per outage day; churn after an unplanned outage; fleet size by age; telemetry cadence and fields. Until then every cost result stays "hypothetical". Use the Elkan rule as a quick check once you have numbers: alert only if the calibrated failure probability exceeds C_visit / (C_visit + C_emergency_failure - C_planned_dispatch).
2. **Back up the data** (raw zips 2.8 GB and generated data about 6 GB are only on the Mac; derived files are in GitHub).
3. **Decide on 6e and 6f.** Recommendation: skip 6e (21 GB, home storage rather than solar home systems; it would only add an external validity check) and do 6f only if you need temperature priors for an essay point.

### Priority B: next 2 to 4 weeks
4. **Finish 6g**: fold Phase 6 into `REPORT.md`, write the operator data request and a **shadow-mode protocol** (log alerts for 2 to 3 months, no dispatch, measure precision at the real failure rate, with the confirmation step).
5. **Do the LFP pack-level question properly.** Pre-register a protocol: fix the labels, the detector and the alarm threshold before looking; hold out systems; add a rest-free resistance estimator from short current steps (dV/dI) and temperature compensation. Also test the cheap hedge: **per-cell daily min/max voltage** (a few extra numbers per day) against the pack-only signal. Seek more labelled LFP data, ideally repair outcomes, from the BattGP authors, the manufacturer or an operator; ten faulty systems cannot settle this.
6. **Lead-acid robustness:** repeat the minimum-data table on all batteries (not 40%), and with location data when available (leave-one-district-out).

### Priority C: 1 to 3 months
7. **Shadow-mode trial with an operator** (see 4). Decide go / no-go with these gates, to be agreed in advance: (a) planned-versus-emergency cost gap large enough that the best policy beats reactive-only at the real base rate (break-even surface in section 3); (b) precision at the real failure rate with confirmation of at least a level that makes the trip economics work (the cost model gives the number once real costs are in); (c) realised false-alarm rate stays within the conformal budget month after month, re-calibrated by age band.
8. **Confirmation step design:** a cheap remote check (SMS or remote load test) is the lever that actually removes false dispatches; quantify its sensitivity and false-confirm rate (assumed 0.90 and 0.10 in 6a).

### Priority D: essay and open specification
9. **What the evidence supports saying now.** Lead-acid: a daily summary record carries usage-based warning of about 70 to 79% balanced accuracy for a 4-week horizon; resistance adds little. False alarms, not accuracy, decide commercial value; recommend a confirmation step and an explicit false-alarm budget set by conformal calibration per age band. LFP: not shown that pack-level data suffices; keep per-cell voltage (or per-cell min/max) in the specification until proven. Licence: the data are CC BY-NC; commercial use needs the operator's own data.
10. **What not to claim:** that pack-only telemetry works for LFP; any saving in currency from the placeholder costs; the as-sampled failure hazard as a fleet failure rate.

### Decisions I need from you
- Share the operator contact or let me draft the data request for you to send.
- Approve or skip 6e (about 21 GB); say whether 6f matters.
- Whether the repo stays private or specific people get access.
- Priority between the essay deliverable (needs only 6g) and further LFP work (needs new data).

---

## 8. Repository map, reproducibility, data
- Code: `sunfix/` (not `src/`: BattGP uses a top-level `src`); configuration `config/costs.yaml` (hypothetical); reports `docs/`, `REPORT.md`, `results/reports/*.pdf`; results `results/`; small derived data `derived/` (tracked); raw data and large generated data are local only (`data/`, zips).
- Reproduce: `make all` (about 35 min) for Phases 1 to 3; individual modules for Phases 4 and 6 (`python -m sunfix.phase4`, `run_6a`, `bench`, `conformal`, `lfp_reproduce`, `lfp_pack_run`, `lfp_pack_degrade`, `lfp_transfer`); PDFs via `python -m sunfix.md2pdf`. Environment: Python 3.12, numpy pinned at 1.26.4.
- Licences: Bboxx and BattGP data CC BY-NC 4.0 (research only; patent application GB 2105995.1 on the Bboxx method); BattGP software BSD-3.
- Every result file named in this report is in `results/`; every methodological choice is dated in `METHODOLOGY.md`; every session is in `LOG.md`.
