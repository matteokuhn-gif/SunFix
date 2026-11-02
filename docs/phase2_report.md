# Phase 2 report: BattGP resistance model on the development subset

Dev subset: 60 batteries (30 failed, 30 alive, seed 0). Code: `sunfix/ocv.py`, `soc.py`, `resistance.py`, `fit_hyper.py`, `phase2.py`. Figures: `results/figures/phase2_*.png`.
Run: `python -m sunfix.fit_hyper ocv; python -m sunfix.fit_hyper nocv; python -m sunfix.phase2`.

## What was built
1. **OCV curve** from the lab GITT file (discharge part, end-of-rest voltages): 13.34 V full to 11.26 V empty, 21.9 Ah measured capacity.
2. **State of charge** by Coulomb counting inside each charging segment (Aitio Eq. 1, 6): start SOC from the rest voltage via the OCV curve, then integrate charge current minus a gassing current. SOC is in Ah, not acid molarity (a constant factor; inputs are normalised, so no effect on the GP). Uncertainty: 10% of each SOC increment plus 0.15 V measurement/OCV noise (Eq. 6A/B).
3. **Resistance observations** `R = (V - OCV) / |I|` on a 5-minute grid, rows with |I| < 0.2 A or OCV above V dropped (Aitio). Noise per row `(sigma_v^2) / I^2`.
4. **Model**: BattGP's `ApproxSpatioTemporalGP` (Wiener-velocity time kernel + SE over operating point on 20 shared k-means basis vectors plus the reference point), used directly, one "cell" per battery. Evaluated at one population reference point (mean current 0.91 A, SOC 15.2 Ah, temperature 28.7 degC). Causal filter only (no smoother), so each estimate uses data up to that time.
5. **Two variants** (your brief's "OCV-free versus OCV-corrected" is not available as a switch, see `docs/battgp_notes.md`):
   - `ocv`: Coulomb-counted SOC plus GITT OCV curve; operating point = (I, SOC, T).
   - `nocv`: no OCV curve and no SOC; OCV approximated by the rest voltage before the segment; operating point = (I, T).

## Deviations from Aitio and BattGP (all deliberate, all to be listed in REPORT.md)
- **Per-segment random offset** (extra hyperparameter `tau`, 0.10 Ohm for `ocv`): all rows in one charging segment share a common resistance error (SOC and OCV bias). Without it the likelihood pushed the time-process noise to the upper bound (30) so that R0 chased each segment. With it the optimum is interior and the log-posterior improved by about 1,700. This is an extension, not in either paper.
- **No half-normal prior on sigma_f** (Aitio Eq. 7). With about 10^4 rows per battery it barely constrains the fit, and their reported sigma units are not comparable to ours. Kept the inverse-gamma prior on lengthscales and bounds on all parameters.
- Hyperparameters fitted on 8 of the 60 dev batteries (4 failed, 4 alive; no labels used), pooled, then frozen. Aitio fitted per battery and took medians. Nelder-Mead, so possibly a local optimum.
- Gassing-current constants are generic VRLA assumptions (I0 = 5 mA at 25 degC and 13.5 V, doubling about every 10 K, factor 10 per 0.29 V); Aitio took theirs from the literature but do not list them.
- Observation noise is heteroskedastic as in Aitio (BattGP's own model is homoscedastic).
- 5-minute grid (every 5th interpolated row) instead of 1 minute, for speed; Aitio's supplement Table 2 reports no loss from 5-minute sampling.
- Not done: Aitio's random-walk benchmark model; to be added in Phase 3 only if we need it.

Fitted hyperparameters (normalised units): `ocv`: sf0 11.6, sf1 1.70, tau 0.10, lengthscales (I, SOC, T) = 3.6, 5.5, 17.7. `nocv`: sf0 8.3, sf1 5.9, tau 0.16, lengthscales (I, T) = 4.0, 5.8. Aitio's medians were (1.94, 2.79, 3.96) for the lengthscales: current is the most important input in both.

## Sanity checks
**Operating-point component** (`phase2_operating_point_slices_ocv.png`): over the 5th-95th percentile of each input, R changes by about 0.67 Ohm with current, 0.16 Ohm with SOC and 0.13 Ohm with temperature (medians over 60 batteries). Current dominates, as in Aitio (their ranges: 1.58, 0.09, 0.16 Ohm).

**Resistance towards failure** (`phase2_trajectories_ocv.png`, `_nocv.png`): the median failed battery sits at 1.0-1.25x its early-life resistance until about 50 days before repair, then climbs to about 1.9x at the repair date (the "knee" of the paper). The median alive battery stays roughly at 1.2-1.4x with no upturn. Individual trajectories are noisy, with occasional spikes to above 1.6 Ohm in both groups (low-information segments), so single-battery readings are much less clear than the medians.

**Separation of the groups** (one number per battery; AUC 0.5 = no separation; 95% bootstrap interval, n = 60, so wide):
| Feature | ocv | nocv |
|---|---|---|
| Slope of R0 over last 60 days | 0.71 (0.55-0.85) | 0.69 (0.55-0.84) |
| R0 in last 30 days | 0.61 (0.46-0.76) | 0.55 (0.40-0.71) |
| Ratio last 30 days / days 60-120 | 0.54 (0.38-0.69) | 0.65 (0.50-0.80) |
| Rise (difference) | 0.57 (0.41-0.73) | 0.65 (0.50-0.79) |
Median slope of the last 60 days: failed +2.6 mOhm/day, alive -0.1 mOhm/day (`ocv`). Plain reading: the slope separates the groups moderately; the resistance level alone does not, because resistance rises in alive batteries too and varies between batteries. That is the same reason Aitio's classifier benefits from combining the level and the slope with stress factors. Whether this reaches 76-82% balanced accuracy is a Phase 3 question; these single numbers are not a classifier.

## Variant comparison (fit quality)
| | ocv | nocv |
|---|---|---|
| Voltage prediction error for the next charging segment (RMSE, mean over 60 batteries) | 0.174 V | 0.235 V |
| Mean squared standardised innovation (1.0 = well calibrated) | 0.60 | 0.82 |
| Log-likelihood per row | 0.59 | 0.52 |
| Seconds per battery | 0.76 | 0.44 |
The `ocv` variant predicts voltage about 26% better. The `nocv` variant gets close on group separation, but with n = 60 the differences between variants in the separation table are within the noise. Recommendation: `ocv` as the main model; keep `nocv` for Phase 4, where "no SOC estimate possible" is one of the data-poor scenarios.

## Runtime
0.76 s per battery for `ocv` (0.44 s for `nocv`) once segments are cached; segment extraction adds about 0.3 s per battery. Projection for all 1,027 batteries: about 13 minutes on one core for the GP, 2-3 minutes with 6 processes. No downsampling or extra parallelisation is needed. Segment cache for all batteries is about 3 GB, well inside the limits.

## Open issues and risks
- Spikes in single trajectories will add noise to Phase 3 features; options are robust medians over a window, or using the posterior uncertainty as a feature (planned).
- Hyperparameters were fitted on a handful of dev batteries; Phase 3 must refit inside each training fold to avoid leaking test batteries (cheap: about a minute).
- The gassing assumptions and the SOC estimate from a single new battery's OCV curve are not validated against aged batteries.
