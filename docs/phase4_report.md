# Phase 4 report: minimum-data experiments

Run: `SUNFIX_SUB_FRAC=0.4 python -m sunfix.phase4 all; SUNFIX_SUB_FRAC=0.4 python -m sunfix.phase4 table` (about 25 min). Outputs: `results/minimum_data_table_sub40.csv` (all scenarios, sets, models, h = 28 and 56), `results/minimum_data_headline_28d_sub40.csv`, `results/figures/phase4_minimum_data_sub40.png`. Full-data (1,027 batteries) results exist only for `interval_5min` and `daily_summary` (`data/phase4/`, git-ignored; see below).

## Design
- **Subsample (decision of 2026-11-03, to save time):** a fixed stratified 40% of batteries (seed 0; 410 batteries, about 196 failed). Same Phase 3 folds restricted to this sample, so no test battery was used in a hyperparameter fit. Consequence: the 95% bootstrap interval is about +-4 points, so differences smaller than that are not detectable.
- Each scenario degrades the cleaned telemetry, then re-runs segment selection, SOC, the resistance GP and the classifiers (GPC and logistic regression, threshold 0.5, 5-fold CV by battery). Decision times are defined from native data, so every scenario predicts the same moments. h = 28 and 56 days.
- Degradations: first sample per interval bin (5 min / 15 min / 1 h; segment gap limit widened to 2.5x the interval); daily min/max/mean of V, I, T plus Wh in and out; temperature removed (GP refitted without T); history window of 30/90/180 days (filter restarted at window start); V rounded to 0.1 / 0.01 V, I to 0.1 A; random row loss of 10/30/50%.
- Native sampling is already 1 minute, so "1 min" is the native scenario.
- **Daily summaries cannot support the resistance GP** (it needs charging curves). That scenario uses a crude proxy, (Vmax-Vmin)/(Imax-Imin), plus usage statistics.
- Per-fold GP hyperparameters are the Phase 3 ones (fitted on native data), not refitted for degraded data, except `no_temperature`. A refit might do slightly better.

## Result at 28 days (balanced accuracy of the full model; 95% interval about +-4 points)
| Scenario | Full model | Usage-only | Gain |
|---|---|---|---|
| Native 1 min | 78.4 | 70.9 | +7.5 |
| 5 min | 79.4 | 70.1 | +9.3 |
| 15 min | 79.2 | 69.7 | +9.5 |
| 1 h | 78.4 | 69.4 | +9.0 |
| Daily summary (proxy) | 79.4 | 70.0 | +9.4 |
| No temperature | 80.7 | 70.7 | +10.0 |
| History 30 / 90 / 180 d | 73.8 / 78.0 / 77.5 | 60.7 / 62.8 / 65.6 | +13 / +15 / +12 |
| V 0.1 V / 0.01 V, I 0.1 A | 77.3 / 78.0 / 78.2 | 71.7 / 71.1 / 70.6 | +6 / +7 / +8 |
| Packet loss 10 / 30 / 50% | 77.2 / 77.9 / 77.4 | 69.4 / 68.5 / 68.3 | +8 / +10 / +9 |

At 56 days all scenarios are 71-76%, and the gain over usage-only is mostly within noise (native: 72.8 vs 72.6).

## What it means
1. **No degradation produced a detectable loss.** Every scenario's interval overlaps native. Thinning to one hourly sample, dropping half the packets, removing temperature or rounding to 0.1 V/0.1 A does not measurably hurt a 4-week warning. This is a statement about what this sample can detect (+-4 points), not proof of zero loss.
2. **Smallest specification tested that still passes 70%: one daily record** (min/max/mean of V, I, T plus Wh in and out): 79.4% at 28 days (interval 75-83), against 70.0% from usage statistics alone. On the full 1,027 batteries it gave 75.1% at 28 days and 70.8% at 56 days (usage-only 68.9%), so treat 75% as the safer number.
3. **The 70% bar is weak** because usage statistics alone sit at 61-72% depending on the scenario. The meaningful claim is the gain over usage-only at 28 days (+6 to +15 points in every scenario). At 56 days the gain is not established.
4. **False alarms** stay high in every scenario: about 180-250 of 1,000 healthy batteries flagged at a decision (threshold 0.5), measured at roughly 50/50 failed/healthy. At realistic failure rates precision is low (Phase 6 quantifies it).

## Limitations
- Resistance-only is weak at short windows (59.5% at 28 d with 30 days of history); the gain with short history comes mainly from the model combination.
- Thinning and loss are emulated from 1-minute data: random loss is not bursty like real outages; thinning keeps the first sample per bin.
- Single-factor tests only. Combinations (for example 15 min + no temperature + 0.1 V + 30% loss) were not run, so "smallest spec" is the best single-factor result, not a tested minimum for a combined cheap device.
- Daily-summary features are computed from the 1-minute data, so a real device with its own clock and sensor would differ.
- One manufacturer, lead-acid, repair decisions as labels.
