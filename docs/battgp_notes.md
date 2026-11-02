# BattGP notes (Schaeffer et al. 2024), read from source at external/BattGP

Commit read: 6d5e1db3337f0de5f3a533acbc09eddccc178e9d. Research software v0.1.0, BSD 3-Clause licence; torch/gpytorch based.

## 1. What it is
Two modes:
- `BattGP_Full`: exact GP, O(n^3), limited to ~40k points on a GPU.
- `BattGP_SpatioTemporal` (the one we want): recursive (Kalman filter + RTS smoother) spatiotemporal GP. Time is handled exactly by a state-space model, the operating point (I, SOC, T) by a basis-vector (inducing-point) approximation. Cost is O(n_steps x n_basis^2..3), linear in the number of rows.

## 2. Model
Resistance is `R0(t, I, SOC, T) = f_time(t) + f_op(I, SOC, T)`, a sum of two independent GPs:
- `f_time`: Wiener-velocity kernel (`src/gp/wiener_kernel_temporal.py`), state = [R, dR/dt], process noise `q = outputscale_wiener`. Non-stationary, so it extrapolates a trend instead of reverting to the prior.
- `f_op`: ScaleKernel(RBF, ARD over 3 dims) with `outputscale_rbf` and `lengthscale_rbf` = (I, SOC, T), approximated on basis vectors (k-means centres of the observed operating points, plus the reference point).
- Observations: `R_obs = (U - OCV(SOC)) / I`, noise `noise_variance` (homoscedastic).
- Output: R0 at a fixed reference operating point (`RefStrategy(Op(I, SOC, T))`, or mean/median), with variance and dR0/dt with variance, on a regular time grid (`sampling_time_sec`). `smooth=True` adds the acausal RTS smoothed trajectory; the causal (filter) trajectory is the one usable online.
- This is structurally the same model as Aitio & Howey Eq. 5 (Wiener velocity + SE over operating point). Differences: homoscedastic noise (Aitio uses per-sample heteroskedastic noise from the SOC uncertainty), hyperparameters fixed (Aitio fits them by MAP), 3 inputs identical.

## 3. Data model and units (this is where our data does not fit)
- `BattData` is hard-wired to **8s1p LFP**: it reads `U_Batt, I_Batt, SOC_Batt`, 8 cell voltages `U_Cell_i`, balancing currents `I_CNV_Cell_i`, 4 temperature sensors, from a Zenodo `field_data.zip` through `read_battery_data`. `_determine_battery_type` returns `(8, 1)` unconditionally.
- Units: current in A, **positive = charging, negative = discharging** (default segment filter `Ibat in [-80, -5]` A, i.e. discharge only; `MODE = "discharge"`). Voltage in V, SOC in %, T in degC, time in days since start (after gap removal), resistance in Ohm.
- **Not OCV-free.** The brief assumed it is. In the code `R = (U - OCV(SOC)) / I`, with `SOC_Batt` supplied by the BMS and `OCV` from `data/ocv_linear_approx.csv` (a linear approximation of LFP OCV vs SOC). The paper's novelty is that the *operating-point dependence* (SOC, T, I) is learned instead of modelled; it still needs an SOC signal and an OCV(SOC) lookup. For our lead-acid data that means we need our own SOC proxy, which is exactly what Aitio build via Coulomb counting plus the GITT OCV curve. A genuinely OCV-free variant would need a different observation model (e.g. regress dU/dI within short windows); that is a research change, not a config switch.
- Cell-level and pack-level models are built separately; fault probabilities compare cells against each other (`fault_probabilities.py`: band around the mean of the other cells, and an absolute threshold). **With one pack voltage and no cells, the cell-comparison fault score does not exist for us.** Only the pack model R0(t) and dR0/dt are usable, plus an absolute threshold probability.
- Name clash: BattGP uses top-level imports `from src.batt_data...`. Our project layout asked for `src/` as well. Two top-level `src` packages cannot coexist on one path.

## 4. Adapting: cheapest path
Skip `BattData` and `BattGP_*`. Reuse the lower layer directly:
`ApproxSpatioTemporalGP` + `WienerTemporalKernel` + `apply_stgp(model, X, Y, ref_op, sampling_time_sec, smooth, max_batch_size)` with
`X = [t_days, I, SOC, T]`, `Y = R_obs` per sample. Basis vectors via `get_basis_vectors("kmeans", ...)` need a `BattData`; we will call k-means ourselves (3 lines). One "cell" = the battery. A 12 V lead-acid battery has a different resistance scale from a single LFP cell (Aitio's figures show R0 varying by up to ~1.6 Ohm with current), so all hyperparameters (noise, outputscales, lengthscales) must be refit; the shipped defaults (`PARAMS_LFP`, e.g. noise 2.33e-6, Wiener 4.2e-13) are for LFP.

## 5. Hyperparameters
`noise_variance, outputscale_wiener, outputscale_rbf, lengthscale_rbf(I, SOC, T)`. Defaults are LFP (config.py). Optimisation exists only for the exact GP (`BattGP_Full.train_hyperparameters`, Adam/LBFGS on a subsample of ~1000-16000 points). `BatteryCellGP_SpatioTemporal.train_hyperparameters` raises `NotImplementedError`. Plan: fit hyperparameters on the exact GP with a subsample (or write a small marginal-likelihood fit via the Kalman innovations), pool across the development batteries, freeze, then run the recursive model on everything. Must be done on training batteries only to avoid leaking test information.

## 6. Quirks to watch
- `predict_cell_r0_op` must be called once per model object (Kalman state is not reset).
- Gap removal (`GAP_REMOVAL_THRESHOLD = 90` days) keeps only the last section after a gap; irrelevant to us until Phase 1 shows gap sizes.
- Negative posterior variances happen numerically and are replaced by the prior variance (`cleanup_negative_variances`).
- `apply_stgp` indexes samples with a window search on `tt` assuming sorted time; needs unique, sorted time stamps.
- `time_step` allocates and stores `z`, `P` per step for smoothing (`smoothing_steps=-1`): memory is O(steps x (nb+2)^2). With 60 basis vectors and hourly steps over 700 days that is ~17k steps x 62^2 x 8 B x 2 ~ 1 GB per battery. Use fewer basis vectors (Aitio use 20), a coarser step, or causal-only (no smoothing) for the full run.
- Runtime complexity: per time step one Kalman update per batch: `O(batch x (nb+2)^2 + nb^3)` for the matrix inverse inside `update` (it inverts a batch x batch matrix, hence `max_batch_size`). Linear in the number of rows.
