# Phase 6d report: LFP transfer on the BattGP field data (in progress)

Data: Zenodo 13715694, `field_data.zip` (1,668,409,464 bytes, MD5 verified against Zenodo's 827f5087...), CC BY-NC 4.0 (research only), downloaded 2026-11-03 to `data/battgp_lfp/` (made read-only; never modified). Contents: 28 CSV files (`data_sys_1..28.csv`), about 19.9 GB uncompressed, 133 million rows per the Zenodo README, 24 V / about 160 Ah LFP systems, 8 cells in series, one current sensor, 9 voltage sensors, 4 temperature sensors, active balancing. All systems were returned to the manufacturer for unsatisfactory behaviour, so the sample is biased (not representative of the fleet). The BattGP cache (feather) and results live in `data/battgp_cache/` and `data/battgp_results/` (git-ignored).

## 6d.1 correctness check against the paper: PASSED on cell identification and timing, with caveats
Paper read: arXiv 2406.19015v3 (the preprint; it says it is outdated, the published version may differ). `python -m sunfix.lfp_reproduce 3 9 6` runs BattGP unchanged (recursive spatiotemporal GP, CPU, 1-hour step). Three settings were run: `repo` (BattGP's gp_runner defaults: fixed reference operating point -15 A / 90% SOC / 25 degC, band 0.55 mOhm, current limit -80 A, SOC up to 95%), `paper` (paper Table 3 criteria: current -200..-5 A, SOC 40-94%, band b = 0.33 mOhm), `paper_meanref` (paper settings with the mean-of-data reference operating point, which the paper describes). One system at a time: system 3 13 s, system 9 36 s (3.7 GB peak), system 6 2.3 min (6.5 GB peak).
What the paper says vs what we get:
| Paper (arXiv v3) | This reproduction |
|---|---|
| System 3: "cell 7 shows a strong upward resistance trend, while other cells have fairly constant resistances" (127-day section) | Cell 7 has by far the largest resistance rise (+1.4 mOhm against 0.2-0.4 for the others, paper settings); same under repo settings (+0.46 vs about 0.1) |
| System 6: "cell 8 ... fault probability increases after 600 days and crosses 0.5 shortly before 1000 days"; small probability at the start | paper_meanref, smoothed: cell 8 about 0.01 at the start, first above 0.5 on day 1020; cell 8 rises +1.1 mOhm vs at most 0.12 for the others |
| System 9: "cell 2 ... knee point after 240 days" (Table 2; its text says day 220 and, confusingly, system 10) | Cell 2 first above 0.5 on day 241 (paper_meanref) / 234-241 (other settings); rise +2.6 to 3.2 mOhm vs at most 0.3 |
Caveats, all material:
- **The reference operating point matters.** With BattGP's default fixed reference point (repo setting), system 6 cell 8 is flagged for the whole period (probability 1.0 from the start), which contradicts the paper's "small at the start"; with the mean-of-data reference it matches. Any pack-only comparison must fix this choice up front.
- **Absolute-threshold probability (2 mOhm) is not in the paper text** I could read, and saturates at 1.0 for every cell and system here because each cell's R0 at the reference point is about 8-9 mOhm. Treat it as an unvalidated repo feature and ignore it.
- **Forward (causal) probabilities have a start-up transient** (about 0.5-1.0 in the first days while the filter variance is large). Onset times must exclude a burn-in (30 days used below).
- Paper's own figure caption and panel titles disagree about system numbers, and the text calls the system with the day-220 knee "system 10"; the Table 2 row for system 9 is used here.
- Hyperparameters: BattGP's shipped LFP values (medians from the paper's runs); the paper tuned on systems 6, 8, 9, 10.
- Paper numbers for causal probabilities, exact curves and the figures were not compared (figures not read).

## Reconciliation
- **Systems, rows:** the paper (arXiv v3) says 29 systems, 232 cells and 131 million rows ("29 battery systems returned to the manufacturer for warranty"). The Zenodo release has **28 systems (28 CSV files)** and says 133 million rows, so the discrepancy is between the paper and the released data (the paper's 29 is probably the count before one system was dropped from the release; this is a guess). BattGP's `gp_runner.py` loops over IDs 1-29. The paper models 21 of its systems ("latest section with no gaps larger than 100 days and at least 600 valid data points"). The "28" in the paper is a count of basis vectors, not systems.
- **Sampling rate:** the paper reports a median measurement interval of 5 s, matching what we measured on systems 3 and 9 (median 5 s; 79-85% of intervals at most 5 s; the rest mostly 60 s; duplicated timestamps 0.1-0.2%). Filter updates use a 1-hour sampling time.
