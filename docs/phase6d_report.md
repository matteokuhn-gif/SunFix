# Phase 6d report: LFP transfer on the BattGP field data (in progress)

Data: Zenodo 13715694, `field_data.zip` (1,668,409,464 bytes, MD5 verified against Zenodo's 827f5087...), CC BY-NC 4.0 (research only), downloaded 2026-11-03 to `data/battgp_lfp/` (made read-only; never modified). Contents: 28 CSV files (`data_sys_1..28.csv`), about 19.9 GB uncompressed, 133 million rows per the Zenodo README, 24 V / about 160 Ah LFP systems, 8 cells in series, one current sensor, 9 voltage sensors, 4 temperature sensors, active balancing. All systems were returned to the manufacturer for unsatisfactory behaviour, so the sample is biased (not representative of the fleet). The BattGP cache (feather) and results live in `data/battgp_cache/` and `data/battgp_results/` (git-ignored).

## 6d.1 correctness check: status = reproduction runs, comparison with the paper NOT yet done
`python -m sunfix.lfp_reproduce 3 9 6` runs BattGP unchanged (recursive spatiotemporal GP, CPU, 1-hour step, shipped LFP hyperparameters, band 0.55 mOhm, upper threshold 2 mOhm, the settings of its `gp_runner.py`). One system at a time, read from the zip: system 3 took 13 s, system 9 36 s (peak memory 3.7 GB), system 6 2.3 min (peak memory 6.5 GB; the largest file, 2.9 GB CSV).
Raw output (smoothed variant), relative "band" fault probability per cell:
- System 3: no clear fault; the largest value is cell 3 (max 0.52, above 0.5 for 67% of the time), other cells 0.31-0.48.
- System 6: **cell 8 flagged for 100% of the time** (max 1.00); cell 3 briefly (max 0.78, 2% of the time); other cells 0.
- System 9: **cell 2 flagged** (max 1.00, above 0.5 for 25% of the time); other cells about 0.
- The absolute-threshold probability is 1.0 for every cell in all three systems (so it does not discriminate under these settings; the units/threshold interpretation needs checking against the paper).
**Not verified:** the paper's own per-cell numbers. The publisher page blocked automated reading (HTTP 403) and no copy of the paper is on this machine. Until the paper (or a legal open copy) is read, "reproduces the paper" is **not** established; the output above is only plausible-looking: relative faults point to single cells in systems 6 and 9, as the method intends.

## Reconciliation
- **28 vs 29 systems:** the zip contains 28 CSV files (systems 1-28) and the Zenodo README says 28. BattGP's `gp_runner.py` loops over IDs `range(1, 30)`, i.e. 1-29, so a "29" most likely comes from that loop bound (an off-by-one in the code, nothing missing from the data). Whether the paper itself says 28 or 29 is unverified.
- **Sampling rate:** measured on systems 3 and 9 (cached data): median interval 5 s; 79-85% of intervals are 5 s or shorter; the rest are mostly 60 s (a slower logging mode), 99th percentile 60 s. Duplicated timestamps are rare (0.1-0.2%). Zenodo's README does not state a rate.
