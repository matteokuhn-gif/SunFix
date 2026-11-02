# Phase 1 EDA and data pipeline report

Figures: `results/figures/eda_*.png`. Code: `sunfix/load.py`, `sunfix/summarise.py`, `sunfix/clean.py`, `sunfix/eda.py`. Numbers below are from all 1,027 batteries unless marked "dev" (60-battery development subset, seed 0, 30 failed + 30 alive).

## What the data is
- 1,027 batteries, 619.5M raw rows (matches the paper's 620M), 183k to 1.04M rows each (median 585k).
- Columns: unix time, current (A), voltage (V), temperature (degC). **Negative current = charging**, confirmed from the data: mean voltage is 13.7 V when I < 0 and 12.7 V when I > 0, and voltage falls when current rises. This matches the supplement's "discharge if I > 0.05 A" and contradicts the main text's "charging positive" (that is their modelling convention).
- 491 failed (entered repair), 536 alive (censored). Lifetime (activation to repair) is 401-761 days for failed and 403-743 for alive; the distributions overlap closely (medians 500 vs 528), as intended by Aitio's lifetime-stratified sampling of healthy batteries.

## Sampling and missing data
- Median sampling interval 60 s for every battery (dev: 88.6% of intervals are exactly 60 s, 95.9% are <= 120 s, 0.3% exceed 610 s; see `eda_distributions.png`). The 99th percentile of the interval is 600-625 s for the median battery, so some batteries send mostly every 10 minutes at times. Temperature is repeated between updates: ~91% of consecutive temperature values are identical, i.e. it is refreshed far less often than V and I.
- No NaNs, no unsorted or duplicate timestamps in any battery.
- Telemetry gaps: within the service window 155 of 1,027 batteries (101 failed, 54 alive) still have at least one gap longer than 30 days, up to 459 days. Aitio's supplement says batteries with gaps over 30 days were removed, so this file does not reflect that filter. **Flagged, kept.** Median fraction of lifetime with gaps over an hour: 2.9%; max 80%.
- Voltage and current are coarsely quantised (about 3,700 distinct voltage values across the dev subset), which matters for the Phase 4 precision tests.

## Ranges
- Current: -5.0 to +5.0 A (sensor limit appears to be 5 A); 99th percentile of charging about 3 A, i.e. at most ~0.25C on a 20 Ah battery. Median battery's 1st-percentile current -1.8 A.
- Voltage: 0 to 20 V raw, but impossible values are rare (dev: ~750 of 35.7M rows below 9 V or above 16 V). Typical mean 13.1 V.
- Temperature: 3.9 to 57.9 degC, no implausible values found.

## Cross-check of telemetry vs. metadata (important)
| Check | Result |
|---|---|
| `Lifetime` = repair date - activation date (failed) | exact for all 491 |
| Telemetry starts after activation | median +0.5 days; 14 batteries start more than 1 day before activation (up to 129 days) |
| Alive: telemetry ends | 503 of 536 end in Oct 2020 (data extraction date); `Lifetime` is ~20 days shorter than the telemetry span for alive batteries |
| **Failed: telemetry ends vs repair date** | **telemetry continues past the repair date for 98.8% of failed batteries, median 89 days, up to 356 days** |
The post-repair stretch looks different: in the 30 days after the repair date the dev batteries send about 8x fewer rows, then normal volume resumes (consistent with a repaired or replaced unit, my interpretation). **Using the full series for failed batteries would put post-repair data into the "before failure" features.** Rule adopted: truncate failed batteries at the repair date.

## Cleaning rules (`sunfix/clean.py`)
1. Keep activation date <= t <= repair date (failed) or end of data (alive). Dev: drops 4.1% of rows, all from failed batteries.
2. Drop NaN rows (none occur).
3. Drop V < 9 V or V > 16 V (physically impossible for a 12 V VRLA; seen: 0 V and up to 20 V).
4. Drop duplicate timestamps (keep first) and sort by time (none occur).
5. Gaps are not removed here; the segment rule below requires no gap > 610 s inside a segment, as Aitio.
No temperature outlier rule was needed. Aitio's step "remove batteries with tampering alerts / bad temperatures" is not reproducible (no alert data), but those were already applied upstream.

## Aitio-style charging segments (the 603k -> 38k reduction)
Charging = I < -0.05 A. A segment is a run of charging samples with no gap > 610 s that satisfies Aitio's Table 1: duration > 6,000 s; the sample before the run (rest state) has V in 11.5-12.9 V and |I| < 0.1 A; max V in the run > 14 V. Each segment is truncated at the first V > 14 V, interpolated to a 1-min grid with PCHIP, and rows with |I| < 0.2 A are dropped. **The "estimated OCV above terminal voltage" filter needs the SOC estimate and is applied in Phase 2.**
Dev result: 328 segments per alive and 304 per failed battery (mean 316; Aitio: 261), 42.6k rows per battery (Aitio: 38k, which includes the OCV filter). Ranges: 118-568 segments. Our definition of a "segment" is slightly more permissive than theirs, which the paper does not define precisely. Segments are cached in `data/segments/<ID>.parquet`.

## Observations to keep in mind
- Failed batteries charge with larger currents on average than alive ones in the example plots (about -0.6 A vs -0.35 A daily mean); usage differs between the groups, so features that describe usage can carry label information (consistent with the stress-only classifier in the paper reaching ~70%).
- Several failed batteries show a voltage drop in the last days before the repair date (`eda_examples.png`). That is a real, if late, signal and supports the h = 0 results.
- The five example series per group are random draws, not hand-picked.
