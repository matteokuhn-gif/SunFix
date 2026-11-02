# Aitio & Howey (2021) protocol, as far as the main paper states it

Sources: Joule 5(12):3204-3220 (PDF in ~/Desktop/SEE) and the arXiv-style PDF `Aitio_and_Howey_2021_predicting_battery_end.pdf` (35 pages, main text plus Supplementary Material, pp. 22-35), added by the user on 2 Nov 2026. Items marked [SI] come from the supplement. Pages 28-29 (Kalman/energy-function equations, Eq. S15) were only skimmed; they matter only if we reimplement their MAP fit, which we do not plan to.

## Data
- 1,027 Bboxx lead-acid batteries (536 healthy, 491 failed), 400-760 days each, 620M rows. Selected so each series is >= 400 days and the classes are about balanced [SI].
- Failure label = battery entered the repair system (customer-driven, so noisy: false positives and negatives acknowledged).
- Paper says charging current is positive. The dataset README given to us says negative = charging. **Verify empirically in Phase 1** (voltage rises when charging).

## Segment selection (Table 1), charging only
| Criterion | Accepted |
|---|---|
| Segment duration | > 6,000 s |
| Starting voltage | 11.5-12.9 V |
| Starting current | < 0.1 A |
| Max voltage in segment | > 14 V |
| Max time gap in recorded data | < 610 s |
Then: truncate each segment to V <= 14 V; interpolate to a 1-min grid with piecewise cubic Hermite; drop samples with I < 0.2 A or with estimated OCV above terminal voltage. Result: ~261 segments and ~38k rows per battery (from ~603k).

## State of charge
Acid concentration c used as SoC. Initial c from the GITT-measured OCV(c) curve (25 degC, one battery) at the minimum-current point; then Coulomb counting with a lumped gassing-current term (Eq. 1; gassing parameters from literature, electrolyte volume from a least-squares fit via Bode). Not available to BattGP, we must rebuild it.

## Resistance model
- Observation: `V = OCV(c) + R0(t, I, T, c) * I + eps` (Eq. 3).
- `R0 = GP(0, k)` with k = Wiener-velocity(t) + SE(I, T, c) (Eq. 5). Inputs I, T, c normalised with population mean and std; time = seconds since start / 34,560,000 (400 days).
- Heteroskedastic noise: `s_n^2 = 0.0225 + Var(c) * (dOCV/dc)^2`, i.e. 150 mV s.d. plus propagated SoC uncertainty (10% uncertainty on dc/dt) (Eq. 6A/B).
- Hyperparameters (two variances, three lengthscales) by MAP with half-normal (s=0.2) priors on variances and inverse-gamma (a=1, b=2) on lengthscales; fitted jointly over the whole population (39M rows, 80 min on 30 Spark cores).
- Recursive/Kalman inference, 20 k-means basis points over (I, T, c).
- Evaluate each battery at **one common reference operating point** = population mean of T, I, c (chosen over per-battery points: 37% lower spread). Gives R0(t) and dR0/dt.
- Benchmark model: same observation model, R0 a random walk in time, no operating-point dependence.

## EOL classifier
- GP classifier (scikit-learn), SE ARD kernel, uniform hyperprior, hyperparameters by marginal likelihood.
- Inputs: R0, dR0/dt, extrapolated from the last observed charging segment to the decision time; plus stress factors [SI]: calendar age, charge throughput, cycle count, cumulative time at float voltage, mean temperature, mean voltage. Inputs normalised as above.
- Four input sets: (1) R0 + dR0/dt, (2) benchmark random-walk R0, (3) R0 + dR0/dt + stress, (4) stress only.
- **Prediction horizons 0 to 56 days before the end of each battery's series.** For failed batteries the series ends at repair; healthy batteries are evaluated the same way at their series end minus h. The label is the battery's class at the end of its series (failed/healthy). So this is "will this series end in a repair", not "time to failure".
- Evaluation: outer 5-fold stratified CV by battery; training sets 48% failed / 52% healthy using data to the end of each series. Test sets: fixed healthy set, with 40%, 60%, 80% of failed batteries sampled randomly (10 repeats each) plus 100% => 3,100 test cases; metrics averaged. "Nested" CV is mentioned but inner-loop details are [SI]/unspecified; in practice hyperparameters come from the marginal likelihood.
- Metric: balanced accuracy = (sensitivity + specificity)/2.
- Reported: stress+R0 gives 82% at h=0 and 73% at h=8 weeks; random-walk benchmark about 66% at 0 and 50% at 8 weeks; stress-only averages 69% over horizons.

## Points to be careful about (potential leakage or optimism in the protocol itself)
1. Healthy batteries are cut at an arbitrary "end of data", failed ones at the repair date; calendar age and throughput at the decision time may carry information about which class a series belongs to purely through how the sets were selected (400-760 days, near-balanced). The stress-only classifier reaching ~69% suggests such effects exist.
2. Hyperparameters of the resistance GP are fitted on the whole population, including test batteries (unsupervised with respect to labels, but not strictly out-of-fold). We will fit these on training folds only and report both if feasible.
3. Repair date is a customer decision, not a measured end of life.

## Our planned deviations (to be recorded as we go)
- Python/PyTorch on one laptop instead of Spark, so hyperparameters will be fitted on a training subset and frozen.
- SI-only definitions will be reconstructed and marked as such.
- Strict out-of-fold resistance hyperparameters.


## Additions from the Supplementary Material
**Sensor quality:** current calibrated to 0.2%; voltage systematic uncertainty 3% (about 0.4 V on a 13 V reading), temperature 2 degC. Voltage precision therefore limits resistance estimation, which is why Phase 4's rounding tests matter.

**Battery selection (S1):** single manufacturer, 20 Ah, activation after 2018-09-01, lifetime > 400 d. Failed = entered repair with a diagnosed *loss of capacity* (other repair reasons excluded). Healthy batteries were drawn by **stratified sampling on lifetime**, so the lifetime distributions of healthy and failed match by construction (this removes the cheapest age leak). Removed: telemetry gaps > 30 d, tampering alerts, bad temperature readings, batteries with < 50 qualifying charging segments, or whose last qualifying segment is > 30 d before series end. Our 1,027 files are already this filtered set.

**Stress factors (S5) [SI]:**
- age = days since activation;
- cumulative time at float = sum of time steps in float, float defined as a period > 600 s with V in 13.4-13.6 V;
- Ah throughput = cumulative absolute Coulomb count;
- mean temperature and mean voltage = expanding window from beginning of life;
- discharge cycle count = number of continuous discharge periods > 600 s with **I > 0.05 A**. So in the raw data **discharge is positive, charging negative**, which matches the dataset note; the main text's "charging positive" is the modelling convention. Phase 1 will still confirm from the data.

**Hyperparameters (S4):** resistance GP hyperparameters were fitted per battery on the *zero-horizon* (longest) data, the population medians were taken, and then used for all horizons. Median values: l_T 3.96, l_I 1.94, l_c 2.79, sigma_f0 2.26, sigma_f1 0.34 (normalised inputs). Useful as starting values for our own fit.

**Reference results, 48% failed test set (S6, Table 1), balanced accuracy %:**
| Days before | R, dR/dt | R, dR/dt + stress | Benchmark | Stress only |
|---|---|---|---|---|
| 0 | 76 | 82 | 66 | 71 |
| 14 | 73 | 80 | 62 | 70 |
| 28 | 67 | 76 | 56 | 71 |
| 42 | 64 | 76 | 51 | 69 |
| 56 | 63 | 73 | 50 | 66 |
At h=0 the best model has TP 389, FP 78, FN 102, TN 458 (sensitivity 79%, specificity 85%). At h=56: TP 363, FP 151, FN 128, TN 385 (sensitivity 74%, specificity 72%). Our "4 weeks" target is the 28-day row (76%); the brief's "8 weeks" is 56 days (73%).

**Their own minimum-data result (S, Table 2), mean accuracy over test cases:**
| Case | Benchmark | R, dR/dt | R, dR/dt + stress | Stress only |
|---|---|---|---|---|
| baseline (1 min, ~260 segments) | 57 | 69 | 77 | 69 |
| 100 segments | 58 | 67 | 76 | 69 |
| 50 segments | 57 | 66 | 75 | 69 |
| 2 / 5 / 10 min interpolation | 58 | 70-71 | 78 | 69-70 |
So within their pipeline, coarser interpolated sampling and fewer segments cost almost nothing. Caveat for Phase 4: this is *interpolation* of already-selected 1-min-capable charging segments (they still required max gaps < 610 s), not real low-rate telemetry. Our experiments should thin the raw data before segment selection.

**Strong baseline to beat honestly:** stress factors alone reach 66-71%. The resistance features add roughly 5-11 points on top. Our 70% "useful" threshold in Phase 4 is therefore barely above what usage statistics alone achieve in this dataset. I will report a stress-only baseline in every Phase 4 row so the value of the resistance estimate is visible.
