# Aitio & Howey (2021) protocol, as far as the main paper states it

Source: Joule 5(12):3204-3220, read from the PDF in ~/Desktop/SEE. **The supplemental information (SI) is not in that PDF**, and one page of the PDF (p. 6 of 18, containing the start of the GP model discussion and Fig. 3) could not be text-extracted. Items marked [SI] are defined only there (health-metric derivative, stress-factor formulas, full confusion matrix, cost function Eq. S15, data selection). Fetching the SI (doi 10.1016/j.joule.2021.11.006) or the arXiv preprint 2107.13856 is the first thing to do before Phase 3.

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
