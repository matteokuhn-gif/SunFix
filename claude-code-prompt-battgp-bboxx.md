# Claude Code prompt: apply BattGP (Schaeffer et al. 2024) to the Bboxx/Oxford field dataset (Aitio & Howey 2021)

Copy everything below the line into Claude Code, opened in `~/Desktop/SunFix_coding`.

---

You are working in `~/Desktop/SunFix_coding` on my Mac (Apple Silicon). I'm building SunFix, software that predicts battery failure in off-grid solar home systems from telemetry the systems already send. I want to apply the open-source **BattGP** library to the **Bboxx field dataset from Aitio & Howey (2021)** and answer two questions:

1. **Reproduction:** using BattGP's Gaussian-process resistance model, how well can we predict battery end of life (EOL) on this dataset at the point of failure, 4 weeks before it and 8 weeks before it? Benchmark: Aitio & Howey report **82% balanced accuracy at failure and 73% eight weeks ahead** (benchmarks 66% and 50%).
2. **Minimum data (my research question):** how much does prediction degrade when telemetry is thinned out (lower sampling frequency, no temperature, shorter history)? I want a table showing the *minimum data a manufacturer must ship* for a useful 30-day warning.

Work in phases. At the end of each phase, stop, summarise what you did and what you found, and wait for me before continuing.

## Context

### Data in this folder (already downloaded; do not re-download)
- `set_0 (1).zip`, `set_1.zip` … `set_10.zip`: 11 zips, 1,027 `.npz` files in total, one per battery, named `<ID>.npz`. Each holds `arr_0`, a float64 array of shape (n, 4). Columns: **[UNIX timestamp, current (A, negative = charging), voltage (V), temperature (°C)]**. Sampling is non-uniform; there are ~600k rows per battery and ~620M rows in total.
- `meta_data.csv` (`meta_data (1).csv` is an identical duplicate; ignore it). Columns: `ID, ACTIVATED (M/D/YYYY), IN_REPAIR_SYSTEM (repair date, empty if none), STILL_ALIVE (TRUE/FALSE), Lifetime (days)`. There are 491 batteries with STILL_ALIVE = FALSE (entered repair, i.e. the failure label) and 536 with TRUE (censored).
- `GITT_OCV.mpt`: BioLogic lab test (tab-separated, header lines first) used to derive the open-circuit voltage curve at 25 °C.
- `zip_to_npy.py`: the authors' helper. **Do not run it as-is**, because it deletes the npz files after converting them. Write your own loader.
- Battery spec: Bboxx 12 V valve-regulated lead-acid (6 cells in series), 20 Ah nominal, 50 Wp panel, 400–760 days in use.
- Licence: CC BY-NC 4.0 (non-commercial research only). The paper also lists a patent application (GB 2105995.1). Keep everything in this repo as research code.

### Papers
- Aitio & Howey (2021), *Joule* 5(12):3204–3220, "Predicting battery end of life from solar off-grid system field data using machine learning". The preprint is https://arxiv.org/abs/2107.13856, and a PDF may also be at `~/Desktop/SEE/Predicting-battery-end-of-life-from-solar-off-grid-system-field-dat_2021_Jou.pdf`. **Read the Methods and Supplementary sections first**, especially data cleaning, OCV/state-of-charge handling, the resistance GP and the EOL-classifier evaluation protocol (how the time points before failure are sampled for failed versus alive batteries). Reproduce their protocol as closely as you can and document every deviation.
- Schaeffer et al. (2024), *Cell Reports Physical Science*, "Gaussian process-based online health monitoring and fault analysis of lithium-ion battery systems from field data" (open access, https://doi.org/10.1016/j.xcrp.2024.102258). Code: **https://github.com/JoachimSchaeffer/BattGP**. Their method is recursive spatiotemporal GPs that separate time-dependent resistance from resistance that depends on the operating point, and it needs no OCV curve. It was built for LFP packs with 8 cells in series and per-cell voltages.

### Key mismatch you must handle
BattGP expects pack and per-cell data from LFP systems. Our data is a single pack-level voltage from lead-acid batteries.
- Treat each battery as a one-"cell" system, or adapt the data model to whatever is cleanest after you've read the BattGP source.
- Check BattGP's current sign convention against ours (ours: negative = charging).
- Decide whether to use BattGP's OCV-free approach or to add the lead-acid OCV curve from `GITT_OCV.mpt`. Try both if it's cheap, because the OCV-free variant is the one that matters for LFP later.

## Project documentation (create in Phase 0, keep up to date in every session)

Create these four files in the repo root before any analysis.

1. **`CLAUDE.md`**: standing instructions that Claude Code loads automatically at the start of every session. It must contain:
   - a short project summary
   - the Rules section from this prompt
   - the session protocol below, copied verbatim
2. **`README.md`**: the current state of the project for anyone opening the folder. Sections:
   - **Purpose** (2–3 sentences)
   - **Status**: current phase, what works, what's next
   - **Quickstart**: environment setup and the one command that reproduces all results
   - **Data**: what the files are, the licence, and what must never be modified
   - **Repo structure**
   - **Key results so far**: headline numbers with a link to the relevant `results/` file
   - **Open questions / known issues**
3. **`METHODOLOGY.md`**: the scientific record. Sections:
   - **Research questions**
   - **Data and cleaning rules**: every rule, with its rationale
   - **Model**: BattGP setup, kernels, hyperparameters, OCV-free versus OCV variants
   - **Labels and evaluation protocol**: the Aitio protocol and every deviation from it
   - **Features**
   - **Metrics**
   - **Minimum-data experiment design**
   - **Assumptions and limitations**
   - **Decisions**: a dated list of each methodological choice, the alternatives considered and why this one was chosen

   Update it whenever a method changes, never retroactively without noting the change.
4. **`LOG.md`**: a chronological, append-only lab notebook. One entry per session, newest at the bottom, in this format:
   ```
   ## YYYY-MM-DD HH:MM – Session N (Phase X)
   **Goal:** …
   **Done:** …
   **Results:** key numbers, with links to files and figures
   **Problems / surprises:** …
   **Decisions:** (also added to METHODOLOGY.md)
   **Next:** …
   **Commit:** <hash>
   ```
   Never edit past entries. Add a correction as a new entry instead.

### Session protocol (put this in CLAUDE.md)

**At the start of every session:**
1. Read `CLAUDE.md`, `README.md`, the last three entries of `LOG.md`, and the parts of `METHODOLOGY.md` relevant to today's work.
2. Run `git status` and `git log -5 --oneline`.
3. Give me a 5-line summary: where we are, what happened last time, open issues, the proposed goal for this session, and anything that contradicts the docs.
4. Open a new `LOG.md` entry with the date, time and goal.

**At the end of every session, or at each phase checkpoint:**
1. Complete the `LOG.md` entry.
2. Update `README.md`: Status, Key results and Open questions.
3. Update `METHODOLOGY.md` if any method, rule or assumption changed, and add a dated Decisions entry.
4. Commit code and documentation together, with a clear message, and record the commit hash in the log.
5. Tell me what changed in the three documents.

If a session ends unexpectedly, the next session must first reconcile the documents with `git log`, then log the gap.

## Phase 0: Setup (stop after)
1. `git init` (if not already a repo); add a `.gitignore` excluding data files (`*.zip`, `*.npz`, `*.npy`, `*.parquet`, `data/`). Create `CLAUDE.md`, `README.md`, `METHODOLOGY.md` and `LOG.md` as specified above and open the first log entry.
2. Create a Python virtual environment (prefer `uv`; otherwise `python3 -m venv .venv`). Clone BattGP into `external/BattGP` and install it plus its dependencies, editable if possible. Pin versions in `requirements.txt`. Use Python 3.11 or 3.12 unless BattGP needs otherwise.
3. Read the BattGP README, its example notebooks and its core modules. Write `docs/battgp_notes.md`: its data model, expected inputs and units, kernels and hyperparameters, how resistance and fault probability are computed, and runtime complexity.
4. Read the Aitio & Howey Methods and Supplementary sections. Write `docs/aitio_protocol.md`: data cleaning, the resistance model, the health metric and the exact EOL classification and evaluation protocol, including lead times, class balancing, metrics and train/test split.

## Phase 1: Data pipeline (stop after)
1. Write `src/load.py` to stream each battery out of the zips **without extracting everything to disk**. Read the npz from the zip into memory, convert to a pandas or polars DataFrame with columns `t, I, V, T`, and cache each battery as Parquet in `data/parquet/<ID>.parquet`.
2. Parse `meta_data.csv` and derive `failed = STILL_ALIVE == FALSE`, `failure_time` (repair date) and `activation_time`. Cross-check that the telemetry time span matches activation, repair and lifetime, and flag inconsistencies.
3. Write an EDA notebook or report covering:
   - sampling-interval distribution
   - missing data
   - current, voltage and temperature ranges
   - rows per battery
   - lifetimes of failed versus alive batteries
   - five example time series for failed and for alive batteries
4. Implement cleaning, following Aitio where specified: remove physically impossible values, de-duplicate timestamps, handle gaps, and apply any downsampling or filtering the paper uses (they reduced ~603k to ~38k rows per battery). Document every rule.
5. **Start with a development subset of 60 batteries** (30 failed, 30 alive, fixed random seed) before scaling. Watch memory: 620M rows in total will not fit in RAM at once.

## Phase 2: BattGP resistance model (stop after)
1. Fit BattGP's recursive GP resistance model to each development battery and extract:
   - the time-dependent resistance trajectory, with uncertainty
   - the operating-point-dependent component
2. Sanity checks:
   - Does resistance rise towards failure for failed batteries?
   - Is it flatter for alive ones?
   - Plot 10 trajectories from each group, aligned to the failure date or the end of data.
3. Compare the variants, OCV-free versus lead-acid OCV-corrected, on fit quality and on how well resistance separates the two groups.
4. Report runtime per battery and the projected runtime for all 1,027. If the full run would take more than about 3 hours, propose and implement parallelisation (joblib/multiprocessing) or a sensible downsampling.

## Phase 3: EOL prediction and benchmark (stop after)
1. Build features from the resistance trajectories at decision time *t* (no look-ahead):
   - current resistance level
   - slope over the last 2, 4 and 8 weeks
   - level relative to the battery's own early-life baseline
   - uncertainty
   - stress factors Aitio uses, such as time at extreme voltage and temperature
2. Implement Aitio's evaluation protocol from `docs/aitio_protocol.md`. Predict "fails within horizon h" for h = 0 (at failure), 4 weeks and 8 weeks. Use battery-level cross-validation with no battery appearing in both train and test, a GP classifier (scikit-learn, as in the paper) plus a logistic-regression or gradient-boosting baseline, and **balanced accuracy** as the headline metric. Also report precision, recall, the false-alarm rate per 1,000 batteries per month, and calibration.
3. Scale to all 1,027 batteries, then produce a results table against Aitio's 82% and 73%.
4. Be honest: if we fall short, analyse why (cleaning, model, protocol) rather than tuning on the test set.

## Phase 4: Minimum-data experiments (stop after)
Re-run Phases 2 and 3 under degraded telemetry and report the 4-week and 8-week balanced accuracy for each:
1. **Sampling interval:** native, 1 min, 5 min, 15 min, 1 h. Also try a daily-summary-only scenario (daily min/max/mean of V, I and T, plus Wh in and out), which is roughly what cheap PAYGo devices send.
2. **Temperature:** with and without.
3. **History length:** last 30, 90 and 180 days, and full history.
4. **Precision:** voltage rounded to 0.1 V and 0.01 V; current rounded to 0.1 A.
5. **Packet loss:** randomly drop 10%, 30% and 50% of rows.

Deliver `results/minimum_data_table.csv` and a chart, then state plainly the **smallest data specification that still gives a useful 30-day warning**, defining "useful" as balanced accuracy of at least 70% and a false-alarm rate I can then decide on. This is the core input for my essay and for the open-data specification (an extension of OpenPAYGo Metrics).

## Phase 5: Write-up
`REPORT.md` (2–3 pages):
- what was done
- reproduction results versus the paper
- the minimum-data table
- limitations: lead-acid, not LFP; one manufacturer; censoring of alive batteries; licence
- next steps for an LFP validation, e.g. running the same pipeline on the BattGP LFP dataset (https://zenodo.org/records/13715694) and later on operator data

Keep all figures in `results/figures/`.

## Rules
- Never modify or delete the original zips or `meta_data.csv`. Write everything new under `data/`, `src/`, `notebooks/`, `results/` and `docs/`.
- No look-ahead leakage: features at time *t* may only use data up to *t*. Split by battery, not by row.
- Fix random seeds and make every result reproducible with one command (`make all` or `python -m src.run_all`).
- Commit after each phase with a clear message. Follow the session protocol: no session or phase ends without updating `LOG.md` and `README.md`, plus `METHODOLOGY.md` when methods changed.
- If something in BattGP doesn't fit our data, explain the problem and the options before making a large workaround.
- Tell me before any step that will take more than 30 minutes or use more than about 16 GB of RAM.
- Explain results in plain language. I'm not a battery engineer; I need to tell an operator, an investor and a ONE Earth reviewer what this shows.

Start with Phase 0.
