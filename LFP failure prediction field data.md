# SunFix must build the first labelled LFP benchmark

Nobody has yet published a labelled, horizon-specific failure predictor for **LFP batteries in the field**. The closest work, Schaeffer et al. (2024, BattGP), carries the Aitio & Howey Gaussian-process (GP) resistance method over to 21 LFP packs. It flags anomalous cells but reports **no accuracy, no false-alarm rate and no lead time**, and its online estimates take **30–150 days to settle** ([arXiv 2406.19015](https://arxiv.org/html/2406.19015v3)). Public LFP field data comes down to two usable sets: BattGP's 28 returned 24 V packs (CC BY-NC) and RWTH's 21 German home-storage systems (CC BY 4.0, mixed chemistry). No public LFP data exists from SHS, telecom or African sites. The commercial risk sits in false positives, not accuracy. Aitio & Howey's 73% balanced accuracy at 8 weeks implies a **28.9% false-positive rate**. At a realistic failure base rate that produces roughly **19 wasted trips per useful one** (worked example below). SunFix's edge therefore lies less in a better classifier and more in cost-optimised thresholds, persistence rules and a cheap remote confirmation step before dispatch. On chemistry, LFP is now **over 55% of EV batteries and over 90% of stationary storage installations (2025)** ([IEA GEVO 2026](https://iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries)). The quality-assured off-grid segment is about **97.5% lithium by sales**, but no public source splits LFP from other lithium chemistries, and none gives a Zambia chemistry split.

Verification key used throughout: **[V]** means the source page was read during the research. **[U]** means unverified (from memory, a search snippet or a secondary summary); check it before citing. **[D]** means derived by arithmetic from a verified source. **[H]** means a hypothetical input for illustration.

## Executive summary

| # | Finding / action | Confidence |
|---|---|---|
| 1 | No published study tests the Aitio & Howey failure classifier on LFP with failure labels. Schaeffer et al. 2024 is the only peer-reviewed GP-resistance study on LFP field data, and it is unsupervised with no performance metrics. | High (within the search scope; the "cited by" list for Aitio & Howey was not exhausted) |
| 2 | The GP-resistance part transfers to flat-OCV LFP: BattGP needs no measured OCV and uses a linear pseudo-OCV plus BMS SOC. Capacity- or OCV-based health estimation does **not** transfer well; Figgener et al. call their LFP results "less reliable". | Medium–high |
| 3 | LFP resistance rises slowly in stationary use (DCR +0.8–1.8 pp/yr vs NMC +4.4–5.2). Failures probably show up as **single-cell knees and imbalance** rather than gradual pack wear, so pack-only telemetry may miss them. This is a hypothesis to test. | Medium |
| 4 | Best public data for SunFix: (1) BattGP LFP field set, (2) RWTH home storage, (3) Sandia LFP subset, (4) HUST 77 cells, (5) MATR (Severson/Attia). Only RWTH is both field data and confirmed for commercial use. | High for 1–2; licences for 3–5 unverified |
| 5 | Benchmark against GP: LightGBM discrete-time hazard; Cox PH and random survival forest; CUSUM/Bayesian change-point on the GP resistance trajectory plus a fleet-relative z-score. Wrap all of them in calibration and a conformal gate. Skip deep sequence models until hundreds of failures are labelled. | Medium (no head-to-head on field data exists) |
| 6 | Balanced accuracy on a 50/50 sample hides the false-positive problem. At a hypothetical 2% monthly failure rate, Aitio-level performance gives about 5% precision. Reaching 50% precision at 75% recall needs **specificity of about 98.5%**. | High (arithmetic) |
| 7 | The cheapest false-positive lever is two-stage confirmation. In Koops's aviation case, cheap confirmation raised net benefit to **97%**, almost regardless of model quality. Persistence rules (alert held for N days) cut false alarms sharply but use up lead time. | Medium (evidence from other domains) |
| 8 | No public figures exist for rural SSA technician-visit cost, SHS battery price or PAYGo loss per outage-day. All cost inputs must come from operators. | High |
| 9 | Chemistry: LFP >55% of EV GWh and >90% of stationary storage (2025, IEA). Off-grid quality-assured SHS about 97.5% lithium by sales (2019–24). Whole SSA market possibly about 70% lead-acid (anecdotal, CGD). Zambia has no chemistry data. | High / High / Low / n.a. |
| 10 | Next step: Phase 6 (below) adds a cost-aware evaluation harness, the benchmark models, and a **pack-voltage-only ablation on BattGP** that tests whether cheap SHS telemetry can detect the cell faults BattGP finds. | n.a. |

## The GP lineage reached LFP but stopped before validation

The verified lineage runs from Aitio & Howey 2021 (lead-acid SHS) to Aitio, Jöst, Sauer & Howey (recursive GP, NCA lab cells), then to Schaeffer et al. 2024 (LFP field data), then Zhou, Aitio & Howey (ageing-aware GP circuit model), then Cornejo et al. 2026 (per-cell recursive GPs in a second-life storage system). Schaeffer et al. cite Aitio & Howey as the model that "motivated" their GP circuit model ([arXiv 2406.19015](https://arxiv.org/html/2406.19015v3)). **No paper reports a labelled, horizon-specific LFP failure classifier** comparable to the 73% at 8 weeks. Aitio & Howey only *claim* their method is "applicable to any chemistry that can be represented with a low order electrical model" ([arXiv 2107.13856](https://arxiv.org/pdf/2107.13856)).

Three findings shape what transfers to LFP. First, resistance estimation survives the flat OCV, because it relies on fast current–voltage dynamics; BattGP uses a linear pseudo-OCV with BMS SOC ([arXiv 2406.19015](https://arxiv.org/html/2406.19015v3)). Zhou et al. warn, however, that OCV errors or drift corrupt the inferred resistance ([arXiv 2407.06639](https://arxiv.org/abs/2407.06639v4)). Second, two lead-acid-specific parts of Aitio & Howey must be replaced: the acid-concentration SOC proxy with gassing parameters, and the assumption that degradation is additive and independent of operating point ([arXiv 2107.13856](https://arxiv.org/pdf/2107.13856)). Third, LFP degrades slowly and locally. In home storage, **LFP DCR grew 0.8–1.8 pp/yr against 4.4–5.2 for NMC and 24.6 for LMO** ([arXiv 2411.08025](https://arxiv.org/pdf/2411.08025)). In BattGP, 9 of 21 systems had one or two deviating cells, and systems 3, 6 and 9 showed knee-shaped single-cell resistance rises. BattGP also cannot tell cell degradation from connector loss or corrosion ([arXiv 2406.19015](https://arxiv.org/html/2406.19015v3)). Per-cell sensing matters: Cornejo et al. find lumped-module models **overestimate SOH by 4–31%** ([arXiv 2609.04487](https://arxiv.org/pdf/2609.04487)). Most SHS LFP packs probably report pack-level data only [U], so SunFix's realistic baseline is a pack-level resistance GP, closer to Aitio & Howey than to BattGP.

### Literature table

Group (a) is stationary/off-grid field data, (b) is EV and vehicle field data, (c) is lab data.

| Paper | Grp | Chem / application | Data | Signals | Method | Target / horizon | Headline result | Code / data | SunFix relevance | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| Aitio & Howey 2021, *Joule* 5(12):3204–20, [10.1016/j.joule.2021.11.006](https://arxiv.org/pdf/2107.13856) | a | Lead-acid 12 V 20 Ah; Bboxx SHS, SSA | 1,027 batteries (491 failed), 400–760 d, about 620 M rows | I, V, T, repair dates | Recursive GP resistance (Kalman filter) plus GP classifier | EOL, 0–56 d | Balanced accuracy 82.4% (0 d), 73.0% (56 d) vs benchmark 66.0% / 50.7% | Data CC BY-NC (Oxford repository); code not public [U] | Baseline being reproduced | V |
| Schaeffer et al. 2024, *Cell Rep. Phys. Sci.*, [10.1016/j.xcrp.2024.102258](https://arxiv.org/html/2406.19015v3) | a? | **LFP** 24 V 160 Ah 8S; RV or solar use (unknown) | 29 systems (232 cells), 21 modelled, about 131 M rows, ≤5 yr, median sampling 5 s | Per-cell V, pack I, 4×T, balancing current | GP circuit model, SE-ARD (I, SOC, T) plus Wiener-velocity time kernel; recursive spatiotemporal GP | Cell and pack fault probability (±0.33 mΩ vs pack mean) | 9/21 systems with outlier cells; forward estimates settle after 30–150 d; **no accuracy or lead time** | [BattGP](https://github.com/JoachimSchaeffer/BattGP); [Zenodo 13715694](https://zenodo.org/records/13715694) | Direct LFP extension; benchmark code | V |
| Figgener et al. 2024, *Nature Energy* 9(11), [10.1038/s41560-024-01620-9](https://ideas.repec.org/a/nat/natene/v9y2024i11d10.1038_s41560-024-01620-9.html) | a | LMO/NMC, NMC, **LFP**; German PV home storage | 21 systems, ≤8 yr | System V, I, P, T at 1 s | Field capacity estimation | SOH (capacity) | Most systems meet warranty ([ess-news](https://www.ess-news.com/2024/09/18/most-home-battery-storage-systems-meet-warranty-claims-new-research-finds/)) | [Zenodo 12091223](https://zenodo.org/records/12091223) | Open, commercially usable PV-coupled data | DOI V; content U |
| Figgener et al. 2024 preprint, [arXiv 2411.08025](https://arxiv.org/pdf/2411.08025) | a | Same | 106 system-years, about 14 bn points | Same | Reconstructed quasi-OCV, ICA/DVA, DCR | Degradation modes | LFP DCR +0.8–1.8 pp/yr; LFP results "less reliable" | Same data | Shows resistance-only LFP signals may be weak | V (peer review not found) |
| Cornejo et al. 2026 preprint, [arXiv 2609.04487](https://arxiv.org/pdf/2609.04487) | a | LMO (not LFP); second-life storage system | 27 modules, 324 cells, 12.5 h | Cell V (10 s), module I/V/T | First-order circuit model with recursive GPs plus EKF | Per-cell SOH, SOC | Lumped model overestimates SOH by 4–31% | [BatteryRecursiveGPs.jl](https://github.com/martincornejo/BatteryRecursiveGPs.jl) | Case for cell-level sensing | V |
| Perriment et al. 2023, IEEE PowerAfrica, [arXiv 2311.14098](https://arxiv.org/abs/2311.14098) | a | Lead-acid SHS | Bboxx field data | Operational | Ageing simulation plus voltage control | Lifetime extension | Simulated +25% life, no loss of load | n.s. | Operator-side complement | V |
| Wang et al. 2023, *Cell Rep. Phys. Sci.* 4(12):101720, [10.1016/j.xcrp.2023.101720](https://pure.bit.edu.cn:443/en/publications/large-scale-field-data-based-battery-aging-prediction-driven-by-s) | b | Chemistry n.s.; EVs | 60 EVs, >4 yr | Usage statistics | Feature screening plus ML with uncertainty | Ageing trajectory, worst units | n.s. on record | n.s. | Fleet-level features can rank the worst units | Metadata V |
| Zhang et al. 2023 (DyAD), *Nat. Commun.* ([DOAJ](https://doaj.org/article/f727a539d2004bba9fce5332a35a3494)) | b | Li-ion; EVs | >690,000 charging snippets, 347 EVs | Charging time series | Dynamical autoencoder | Fault / anomaly | Beats existing detectors; lowers expected fault and inspection cost (no AUROC read) | n.s. | Cost-aware anomaly detection on a fleet | Abstract V; DOI U |
| Voronov et al. 2020, *IJPHM* 11(1), [10.36001/ijphm.2020.v11i1.2608](https://www.papers.phmsociety.org/index.php/ijphm/article/view/2608) | b | Lead-acid; heavy vehicles | Sparse, irregular readouts | Workshop / over-the-air data | RSF vs LSTM, imputation | Time to failure | No numbers on abstract page | n.s. | Survival precedent for sparse telemetry | V |
| Nakano et al. 2024, *Adv. Appl. Energy* 16:100188 ([DOAJ](https://doaj.org/article/353ccf4394314ef491143dcf48081ef2)) | b | Li-ion; EVs | 3 vehicles, about 3 yr | Trip time series | Transformer, self-supervised | SOH | MAE 0.72%, RMSE 1.17%; holds with 50% of labels | n.s. | Label efficiency; tiny sample | Metadata V |
| He et al. 2022, EVBattery, [arXiv 2201.12358](https://huggingface.co/papers/2201.12358) | b | Li-ion; EVs, 3 makers | Hundreds of EVs | Charging records | Deep-learning benchmarks | Health / capacity | Dataset paper | Data public [U] | Possible pre-training corpus | Abstract V |
| Zhao et al. 2025 preprint, [arXiv 2505.05364](https://arxiv.org/pdf/2505.05364) | b/c | Li-ion | Field mapped to lab | Two-point features | Random forest | Capacity, remaining life | Capacity MAPE 1.17–3.08%; remaining life MAPE 12–31% | n.s. | Lifetime is much harder than capacity | V |
| "SOH estimation for LiFePO4 battery system on real-world EVs…" ([BIT](https://pure.bit.edu.cn/en/publications/state-of-health-estimation-for-lifeposub4subbattery-system-on-rea/fingerprints)) | b | **LFP** EVs | ? | ? | ? | SOH | ? | ? | Lead only | **U** |
| "SOC and capacity estimation for MWh-scale LiFePO4 system…" ([BUAA](https://research.buaa.edu.cn/en/publications/state-of-charge-and-capacity-estimation-for-mwh-scale-lifeposub4s/)) | a | **LFP** grid storage | ? | ? | ? | SOC, capacity | ? | ? | Lead only | **U** |
| Aitio, Jöst, Sauer & Howey, [arXiv 2304.13666](https://arxiv.org/pdf/2304.13666) | c | NCA/Si (not LFP); van drive cycle | Lab | V, I, T | Recursive GP for parameter dynamics | Capacity, R0 | Capacity error 0.016 Ah; R0 1.7 mΩ | Data public | Theory behind BattGP | V (venue U) |
| Zhou, Aitio & Howey, [arXiv 2407.06639](https://arxiv.org/abs/2407.06639v4) | c | Li-ion | 2 datasets | V, I | Ageing-aware GP circuit model | Capacity, degradation modes | Relative RMSE <1%, MAPE <2% | n.s. | OCV accuracy needed | V (venue U) |
| Zhou & Howey, [arXiv 2211.05697](https://arxiv.org/pdf/2211.05697) | c | **LFP** (Severson + Attia) | 124 + 45 cells, 61 conditions | Lab cycling | Hierarchical Bayesian linear model | Cycle life | MAPE 8.6%; about 12–13% better than ridge | Public data | Partial pooling for small fleets | V (RMSE unit doubtful) |
| Severson et al. 2019, *Nat. Energy* 4:383–91, 10.1038/s41560-019-0356-8 | c | **LFP** 18650, fast charge | 124 cells | Discharge curves | Elastic net on ΔQ(V) | Cycle life | About 9.1% test error [U] | Public | Standard benchmark | **U** |

A review (Sulzer et al. 2021, *Joule*, DOI 10.1016/j.joule.2021.06.005 [U]) finds that field prediction suffers from uncontrolled conditions, poor sensors and rare validation checks, and recommends hybrid physics–ML approaches ([Oxford Engineering](https://eng.ox.ac.uk/news/predicting-battery-lifetime-from-field-data)).

## Only two public LFP field datasets exist, and only one allows commercial use

All other public LFP data are lab cells cycled at 25–30 °C under fast-charge or drive-cycle protocols. That is the opposite of SHS use: slow and irregular PV charging, partial cycling, enclosures at 35–45 °C and long dwell times. A 2026 review confirms that EV, grid and other field data are "severely underrepresented" ([arXiv 2608.26111](https://arxiv.org/pdf/2608.26111)). "Link resolves" below means the record page was fetched; binary files were not test-downloaded because of proxy limits.

| Rank | Dataset | Lab / field | Chemistry, form | Units | Signals, sampling | Labels | Size | Licence (commercial?) | URL | SHS similarity |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | BattGP field data (Schaeffer et al.) | Field | LFP, 8S prismatic, 24 V about 160 Ah | 28 packs (paper: 29), 1 mo–5 yr, about 133 M rows | Pack I, per-cell V, 4×T, balancing; median 5 s | None; all warranty returns (biased) | 1.7 GB | **CC BY-NC 4.0 (no)** | [Zenodo 13715694](https://zenodo.org/records/13715694) [V] | High: pack format and probable solar use; temperature unknown |
| **2** | RWTH ISEA home storage (Figgener et al.) | Field | LMO/NMC, NMC, **LFP** (LFP count unconfirmed) | 21 systems, 106 system-years, about 14 bn points | System V, I, P, housing and room T; 1 s | Capacity estimates; no failures | about 21.1 GB | **CC BY 4.0 (yes)** | [Zenodo 12091223](https://zenodo.org/records/12091223) [V] | Medium-high: real PV cycling; indoor German climate; system-level only |
| **3** | Sandia (Preger et al. 2020) LFP subset | Lab | LFP (A123 18650 [U]) among NCA/NMC; about 61 cells total | 15/25/35 °C; DoD 0–100/20–80/40–60% [U] | V, I, T, capacity | Capacity fade to and beyond 80% | U | **U** | [Sandia record](https://sandia.gov/research/publications/details/degradation-of-commercial-lithium-ion-cells-as-a-function-of-chemistry-and-2020-01-09); hosted on [batteryarchive.org](https://www.batteryarchive.org/) [S/U] | Best lab proxy for heat and partial cycling |
| **4** | HUST (Ma et al. 2022) | Lab | LFP/graphite 18650, 1.1 Ah | 77 cells | V, I, T; 77 discharge protocols, 30 °C | Cycle life to EOL [U] | U | **U** (Mendeley default often CC BY [U]) | [Mendeley nsc7hnsg4s](https://data.mendeley.com/datasets/compare/nsc7hnsg4s) [S] | Low-medium: varied discharge, full DoD |
| **5** | MATR: Severson 2019 + Attia 2020 | Lab | LFP/graphite 18650 (A123 [U]) | 124 + about 45–192 cells (180 in BatteryML) | V, I, T, IR, capacity; about 10 s processed | Cycle life (EOL 80%); OSBAD anomaly labels on 46 cells | U (OSBAD SQLite about 364 MB) | **U** (believed CC BY 4.0) | [data.matr.io](https://data.matr.io/1/) [U]; `batteryml download MATR` ([BatteryML](https://arxiv.org/pdf/2310.14714)) | Low: fast charge, 30 °C; but the literature benchmark |
| 6 | Turku UAS randomised profiles | Lab | 2 LFP 26650 3.0 Ah (plus NCA/NMC) | 2 LFP cells, >600 cycles | V, I, T at 1 Hz; HPPC every 20 cycles | SOH from reference tests | U | Dataset licence not stated | [PMC12175228](https://pmc.ncbi.nlm.nih.gov/articles/PMC12175228/) [V] | Irregular load; too few cells |
| 7 | CALCE A123 / K2 | Lab | LFP 26650 | 3–5 cells | EV drive cycles | SOC-oriented | Small | **All rights reserved (unclear)** | [CALCE](https://web.calce.umd.edu/batteries/data/) [V] | Low |
| 8 | Empa Aurora | Lab | NMC and LFP coin cells | 199 | BDF time series | 1,000 cycles | 2.5 GB | CC BY 4.0 | [Zenodo 15481956](https://zenodo.org/records/15481956) [V] | Very low |
| ref | Bboxx SHS (Aitio & Howey) | Field | **Lead-acid** 12 V 20 Ah | 1,027 | I, V, T | Repair dates (491) | 49 GB raw | CC BY-NC 4.0 | [Oxford ORA](https://ora.ox.ac.uk/objects/uuid:e41d3d4c-f74e-4d76-81fd-0caa77ec6cec) [S] | Real SSA SHS usage; wrong chemistry |

Contradiction: Zenodo lists 28 systems and about 133 M rows, while the paper lists 29 systems and about 131 M rows. Check the README. Licence implications: BattGP and Bboxx data are restricted to non-commercial research, so they cannot train a product model without written permission. Email Schaeffer (TU Darmstadt/MIT) and the Bboxx/Oxford data owners. Ask RWTH ISEA (Figgener) how many systems are LFP. No public LFP data was found from SHS, telecom, e-bus or e-bike fleets. This is probably a real absence, which makes Zambia pilot data SunFix's main asset.

## Survival models and change-point detection deserve benchmark slots

No paper benchmarks BattGP against other ML methods, and no head-to-head on the same field dataset was found ([arXiv 2406.19015](https://arxiv.org/html/2406.19015v3)). A SunFix benchmark would therefore be novel in its own right. Field evidence is thin: GP resistance tracking (Aitio & Howey; BattGP), deep autoencoders on EV fleets (DyAD), random survival forests and LSTMs on sparse lead-acid readouts ([Voronov et al. 2020](https://www.papers.phmsociety.org/index.php/ijphm/article/view/2608)), and random forests mapping field to lab data. Zhao et al. show that capacity is far easier to predict than remaining life, with MAPE of 1–3% against 12–31% ([arXiv 2505.05364](https://arxiv.org/pdf/2505.05364)). Survival methods (Cox, DeepSurv, DeepHit, MTLR) have been compared only on lab data, using C-index, time-dependent AUC and integrated Brier score ([Xue et al. 2025](https://arxiv.org/html/2503.13558v7)).

| Family | Censoring (alive batteries) | Irregular / sparse sampling | Few failures | Interpretability | Verdict |
|---|---|---|---|---|---|
| GP resistance (baseline) | Not needed (trend plus threshold) | Native (Kalman filter) | Good, but 30–150 d to settle | High | Keep as baseline and feature source |
| **LightGBM/XGBoost discrete-time hazard on landmark features** | Yes, via person-period / landmark framing | Via window aggregation | Works with dozens of events | Medium (SHAP) | **Benchmark 1**: cheap, strong on tabular data. No verified battery failure-ahead metrics yet |
| **Cox PH + random survival forest** (scikit-survival) | Native | Via landmark features | Cox good with few events; RSF needs more | Cox high | **Benchmark 2**: precedent in Voronov 2020; proper censored metrics |
| **Change-point / anomaly: CUSUM or BOCPD on GP resistance, plus fleet-relative z-score** | Not needed | Fine on daily aggregates | No labels needed | High | **Benchmark 3**: catches knees; replaces BattGP's cell-vs-pack comparison when only pack data exist |
| **Calibration + conformal gate**, optional hierarchical Bayesian pooling by district / model | Conformal survival variants exist [U] | Wrapper | Needs a calibration set; pooling helps small groups (about 12–13% gain vs ridge on lab data, [Zhou & Howey](https://arxiv.org/pdf/2211.05697)) | High | **Benchmark 4 / layer**: makes thresholds defensible ([Chen 2013](https://papers.phmsociety.org/index.php/phmconf/article/view/2249)) |
| DeepSurv / DeepHit | Native | Needs fixed inputs | Poor | Low | Defer |
| LSTM / TCN / transformer | Only with survival loss | Needs resampling | Poor; transformer used only 3 vehicles (Nakano) | Low | Defer until hundreds of failures |
| Lead-acid→LFP transfer learning, PINNs | n/a | n/a | n/a | n/a | No field-validated evidence found; test via Phase 6 rather than adopt |

Recommended feature set per system and landmark date: GP resistance mean and variance; 7-, 30- and 90-day slopes; curvature (a knee indicator) and a CUSUM statistic. Add capacity proxies: Ah between voltage thresholds on partial charges, end-of-discharge voltage at a matched load, and time to absorption voltage. Add stress factors: equivalent full cycles, time at high or low SOC, low-voltage-disconnect trips, and time above 35/40 °C. Finally, add data-quality features: gaps and communication outages, since sparsity may itself predict failure.

## False positives are an operating-point problem first

The evidence shows that **the cost and resolvability of a false alarm drive net benefit more than model quality does**. Koops (2018) models expected cost per asset as C = C₀ + C_TP·p(TP) + C_FP·p(FP) + C_FN·p(FN) and picks the operating point by sweeping the ROC curve. He finds that "the highest parameter influence on the achievable net benefit" comes from false alarms ([Koops 2018](https://papers.phmsociety.org/index.php/phme/article/download/460/phmec_18_460)). In his engine case, an expensive false alarm puts the optimum at FPR 0.15–0.21 and TPR 0.84–0.93, with 53–74% relative net benefit. A **$0.2k borescope check** instead of a $2k shop visit moves the optimum to FPR 0.73 and TPR 1.00, and net benefit to **97.4%** ([Koops 2018](https://www.bauhaus-luftfahrt.net/fileadmin/user_upload/Publikationen/2018_Koops_Business_Case_Analysis_Predictive_Maintenance_PHME.pdf)). The two published versions disagree on some table values, so treat the numbers as illustrative. The SHS analogue of the cheap check is a remote confirmation: a PAYGo remote load test, an SMS/USSD customer query, or bundling the visit into a scheduled collection round. That analogy is a hypothesis; no off-grid validation was found.

### Why balanced accuracy misleads

| Quantity (Aitio & Howey, 56 d) | Value | Basis |
|---|---|---|
| TP / FP / FN / TN | 368 / 155 / 123 / 381 | [V] ([arXiv 2107.13856](https://arxiv.org/pdf/2107.13856)) |
| Recall / specificity / FPR | 74.9% / 71.1% / 28.9% | [D] |
| Precision on the 48% failed sample | 70.4% | [D] |
| Precision at 2% monthly failure rate [H] | **≈5%**: 15 true vs 283 false dispatches per 1,000 systems/month | [D] from [H] |
| Specificity needed for 50% precision at 75% recall, 2% prevalence [H] | **≈98.5%** (FPR ≈1.5%) | [D] |

Aitio & Howey also note label noise: "there could be false positives and false negatives since repairs are driven by customer decisions" ([arXiv 2107.13856](https://arxiv.org/pdf/2107.13856)). BattGP's alarms can be real non-battery faults such as connector loss or corrosion ([arXiv 2406.19015](https://arxiv.org/html/2406.19015v3)). Visit outcomes should therefore be labelled {battery fault, other fault, no fault}, not as binary.

### Levers, ranked by expected impact

| Lever | Evidence | Expected effect / caveat |
|---|---|---|
| **Two-stage confirmation** (remote test or bundled visit before a dedicated trip) | Koops Scenario 6 [V] | Largest; makes high-recall settings affordable. Unvalidated in SHS |
| **Cost-optimal threshold** p* = C_FP / (C_FP + C_FN) on calibrated probabilities | Elkan 2001 ([mlanthology](https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations)) [U]; Koops ROC sweep [V] | Needs operator costs; only valid if probabilities are calibrated |
| **Calibration** (Platt for small sets, isotonic for large) | Niculescu-Mizil & Caruana 2005 ([entry](https://mlanthology.org/icml/2005/niculescumizil2005icml-predicting)) [U] | Boosted trees are often miscalibrated; Aitio & Howey report no calibration check [V] |
| **Persistence rule** (N consecutive days) | Adnan et al. 2011: n=1 → FAR 25.3%, delay 0.2; n=3 → 6.4%, 2.9; n=5 → 1.0%, 7.7 samples ([ACC 2011](https://skoge.folk.ntnu.no/prost/proceedings/acc11/data/papers/1262.pdf)) [V] | Assumes i.i.d. samples; autocorrelated daily scores give smaller gains. Spends lead time |
| **Alert budget / precision-at-K** | Design inference | Dispatch min(K, #{p > p*}) ranked by p; ties the model to technician capacity |
| **Fleet-relative baseline** | BattGP's cell-vs-pack ±0.33 mΩ band [V] | Cancels temperature and usage effects; for pack-only data, compare against a peer cohort (same model, install month, district) |
| **Conformal / uncertainty gate** | Finite-sample FPR control under exchangeability ([arXiv 2605.13642](https://arxiv.org/pdf/2605.13642)) [V existence] | Seasonality and drift break exchangeability; use adaptive variants |
| **Ensemble agreement** (2 of 3 models agree) | No battery evidence found | Test it; do not assume it works |

### Evaluation design: expected cost per 1,000 systems per month

E[C] = n_alert·C_check + FP_d·C_visit + TP_d·(C_visit + C_batt,planned) + FN·(C_visit + C_batt,emergency + C_outage·D_outage + C_churn)

Compare against three baselines: (i) reactive only; (ii) age-based replacement; (iii) all alerts bundled into routine visits. Optimise over the threshold τ, persistence N ∈ {1, 3, 5, 7} days, confirmation on or off, and capacity K. Use temporal and district splits, never random row splits. Sample landmark dates across all system-days so the test set has realistic prevalence. Run sensitivity analysis at ±50% on each cost and on the base rate, and report the break-even region. Before acting on alerts, run a **2–3-month shadow mode** in which alerts are logged but not acted on, to measure false positives without bias. Report: E[C] and saving vs reactive; precision and recall at the operating point; **false dispatches per 1,000 systems/month**; **missed failures per 1,000/month**; lead-time distribution (median, 10th percentile, share ≥30 d); Brier score and ECE; alert flips per system-month; bootstrap confidence intervals.

Demand-side stakes justify a churn term. In Uganda, **a one-week increase in average service wait cut SHS adoption by up to 32.4%** ([Kundu & Ramdas 2022](https://ideas.repec.org/a/inm/ormsom/v24y2022i3p1329-1348.html)). No public figure exists for rural visit cost, battery replacement cost or repayment loss per outage-day. Searches of SolarAid, Lighting Global and Hystra pages found none ([SolarAid 2024](https://solar-aid.org/news/launching-the-state-of-repair-report/); [Hystra 2019](https://www.hystra.com/our-insights/pricing-quality-cost-drivers-and-value-add-in-the-off-grid-solar-sector-2019)). The $0.30/system/month SunFix price therefore needs testing against operator figures.

### Operator data request

| Category | Fields |
|---|---|
| Telemetry | System ID, timestamp, V, I, SOC, T, sampling interval; per-cell V if the BMS exposes it; BMS fault and balancing flags; firmware |
| Asset | Chemistry, capacity, cell supplier, install date, product model, district, GPS or distance to depot |
| Failure / repair | Date reported, date diagnosed, fault code, part replaced, technician notes, warranty claims, swap and deactivation records; outcome {battery / other / none} |
| Costs | **Cost per technician trip** by distance band; technician capacity (visits/month); **battery unit price plus logistics** (lead-acid vs LFP); remote-test or SMS cost; routine-visit schedule |
| Commercial | Daily PAYGo instalment; repayment during and after outages (**outage cost**); default and churn uplift after failure; referral data |
| Population | Active fleet by month and age (for monthly hazard and base rate) |

## LFP dominates new lithium, yet off-grid splits remain unpublished

The weighting basis differs between sources and is stated in each row. Do not mix bases.

**Global (2025 latest)**

| Metric | Value | Basis | Source (date) |
|---|---|---|---|
| EV battery deployment | 1.2 TWh, +~30% y/y | GWh | [IEA GEVO 2026](https://iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries) (2026) |
| LFP share of EV batteries | >55% (2025); ~50% (2024); >40% (2023); "much smaller" (2020) | GWh | IEA GEVO 2026; 2023 figure via [Battery Technology](https://www.batterytechonline.com/ev-batteries/iea-report-ev-battery-prices-drop-lfp-surges-sodium-ion-on-horizon) (secondary summary of GEVO 2024, Jun 2024) |
| LFP share of stationary storage installations | **>90%** | Installations | IEA GEVO 2026 |
| LFP outside China | Nickel chemistries ~80% of EV batteries; EU LFP >10%; emerging and developing economies about two-thirds LFP | GWh | IEA GEVO 2026 |
| Alternative EV series | LFP 32% (2023) → 40% (2024) | GWh, passenger EVs | [Adamas](https://www.adamasintel.com/lfp-now-commands-40-of-the-global-ev-battery-market/) (Mar 2025) |
| LFP share of cathode shipments | ~72% (3.47 of 4.95 Mt), +8 pp y/y | **Tonnes** (overstates LFP relative to GWh) | [SNE Research](https://sneresearch.com/en/insight/release_view/681/page/0) (Jun 2026) |
| Total Li-ion demand | 1.59 TWh, +29%; EVs 75%; LFP demand +48% | GWh | [Benchmark](https://source.benchmarkminerals.com/article/global-lithium-ion-battery-demand-rose-29-in-2025) (Jan 2026) |
| Li-ion deployment growth 2020→2025 | >6× | GWh | IEA GEVO 2026 |

The two EV series disagree: Adamas reports 32%/40% and the IEA >40%/~50% for 2023/2024. This is probably a scope difference, since Adamas covers passenger EVs and the IEA includes buses and trucks. Use the IEA as the headline and Adamas as a lower bound. A blended EV-plus-storage LFP share of about 60–65% in 2025 is an inference, not a published figure. IEA shares for 2020–22 and BNEF splits were not retrieved.

**Off-grid solar, sub-Saharan Africa**

| Metric | Value | Basis | Source (date) |
|---|---|---|---|
| VeraSol-certified Solar Energy Kits using lead-acid | 2 of 193 (≈99% lithium) | **Product count** | [CGD Policy Paper 383](https://www.cgdev.org/sites/default/files/lead-safe-off-grid-electrification-understanding-issues.pdf) (Feb 2026) |
| GOGLA-affiliated sales using lead-acid, 2019–24 | 3% (0–10 Wp), 0% (11–49 Wp), 2% (≥50 Wp); ≈2.5% overall | **Sales-weighted** (units); overall [D] from CGD table | CGD 2026 |
| Quality-assured share of SSA market | ≈⅓ | Sales | CGD 2026 |
| Unaffiliated segment lead-acid | Assumed 95% (cites 100%); Malawi 0% lithium | Sales, assumption | CGD 2026 |
| Whole SSA market lead-acid | ~70% (**anecdotal**; "no reliable data") | Sales | CGD 2026 |
| New mini-grids using lithium, 2018–21 | 69% (lead-acid 97% before 2017) | Installations | CGD 2026 |
| LFP vs NMC within off-grid lithium | **No public split found** | — | — |
| Product evidence | Sun King Home 40Z: LiFePO4 ([E4C, Dec 2024](https://www.engineeringforchange.org/solutions/product/sun-king-home-40z/)); Bboxx bPower: "lithium", subtype n.s. ([Bboxx](https://bboxx.com/products/bpower80-to-240-range)) | Product sheets | — |
| GOGLA member kit sales 2025 | ~10 million | Units | [SolarQuarter](https://solarquarter.com/2026/06/09/gogla-members-achieve-record-10-million-off-grid-solar-kit-sales-in-2025-expanding-energy-access-worldwide-report/) (Jun 2026; secondary) [U] |

The best defensible proxy for SunFix's addressable market (investor-backed PAYGo operators) is about **97–99% lithium**. Within that, LFP probably dominates, given the Sun King product sheet, the Chinese cell supply chain, and LFP's cycle life for 2–5-year contracts, but this is unquantified (confidence: medium-low). The installed base still contains lead-acid, both pre-2019 and unbranded. Capacity-weighted (Wh) off-grid shares do not exist publicly. To fill the gap, count chemistry fields in the VeraSol database by hand (blocked in this session) and ask the operators and others named in the brief (RDG, d.light, Ignite, Fenix, Vitalite) to confirm their chemistry.

**Zambia:** no chemistry split exists. NEAS 2023 shows **8.9% of households use SHS** (11.4% rural, 5.5% urban), with a median system of 30 W and a weighted mean of ~166 W. 13.6% of SHS households report battery use. Customs duty and VAT on solar equipment have been waived, and no comprehensive price or quality data exist ([IGC, Sep 2025](https://www.theigc.org/sites/default/files/2025-07/Hardie-et-al-Policy-note-July-2025.pdf)). A median of 30 W falls in the 11–49 Wp band, where GOGLA-affiliate lead-acid share is ~0%. That suggests branded Zambian SHS are lithium, but this is an inference. Next, request GOGLA country data and check whether CGD's Table 3 covers Zambia.

## Phase 6: ready to paste into Claude Code

```markdown
## Phase 6: False-positive economics, benchmark models and LFP transfer (stop after each sub-phase)

Context: Phases 0–5 reproduced Aitio & Howey with BattGP on the Bboxx lead-acid data and produced the
minimum-data table. Phase 6 makes the evaluation cost-aware, adds benchmark models and tests transfer to
LFP. Follow the session protocol in CLAUDE.md. Add a METHODOLOGY.md "Phase 6" section and a dated
Decisions entry for every choice. All public LFP datasets used here are research-only unless stated;
record each dataset's licence in README.md > Data.

### 6a. Cost-aware evaluation harness (Bboxx)
1. Create `config/costs.yaml` with PLACEHOLDER values clearly marked "hypothetical – replace with operator
   data": C_visit, C_check (remote test/SMS), C_batt_planned, C_batt_emergency, C_outage_per_day,
   D_outage_days, C_churn, technician capacity K per 1,000 systems/month.
2. Write `src/eval_ops.py`:
   - Landmark sampling over ALL system-days (realistic prevalence), not a 50/50 sample. Report the empirical
     monthly failure hazard by battery age.
   - Temporal split (train on earlier activation cohorts, test on later ones) plus a leave-one-district-out
     split if location metadata exist. Never split by row.
   - Calibration: Platt and isotonic on a held-out calibration fold; reliability diagram, Brier score, ECE.
   - Grid search over threshold τ × persistence N∈{1,3,5,7} days × confirmation {off, on with assumed
     sensitivity s_c and cost C_check} × capacity K. Output E[C] per 1,000 systems/month, false dispatches
     and missed failures per 1,000/month, precision, recall, lead-time distribution (median, p10, share ≥30 d),
     alert flips per system-month, bootstrap 95% CIs.
   - Baselines: reactive-only, age-based replacement, and Elkan threshold p* = C_FP/(C_FP+C_FN).
   - ±50% sensitivity on every cost and on the base rate; plot the break-even region.
3. Re-express the Phase 3 result in these terms. Deliver `results/ops_metrics_bboxx.csv`,
   `results/figures/cost_vs_threshold.png`, `results/figures/leadtime_hist.png`. Stop.

### 6b. Benchmark models on the same landmarks and features
1. Build `src/features.py` (no look-ahead): GP resistance mean, variance, 7/30/90-day slope, curvature,
   CUSUM statistic; partial-charge Ah between fixed voltage thresholds; end-of-discharge V at matched load;
   equivalent full cycles; time at low/high SOC and LVD trips; time above 35/40 °C; data gaps and comms
   outages; fleet-relative z-score of resistance vs a peer cohort (same install month ± 1).
2. Models: (i) the existing GP classifier; (ii) LightGBM discrete-time hazard ("fails within 30/56 d");
   (iii) Cox PH and random survival forest (scikit-survival), handling censored alive batteries properly;
   (iv) unsupervised CUSUM / Bayesian online change-point on the GP resistance trajectory plus the fleet
   z-score; (v) a 2-of-3 agreement ensemble.
3. Report balanced accuracy (for comparability with 82%/73%), C-index, time-dependent AUC, integrated
   Brier score AND the 6a operational metrics at each model's cost-optimal operating point. SHAP for
   LightGBM. Deliver `results/benchmark_table.csv`. Stop.

### 6c. Conformal gate and persistence on autocorrelated scores
1. Split-conformal (and an adaptive/online variant) calibrated on healthy system-days. Measure the
   realised false-alarm rate on later months to quantify drift and seasonality.
2. Measure the real (autocorrelated) effect of persistence N on FPR and lead time. Compare with i.i.d.
   theory (Adnan et al. 2011: n=1 → 25.3%, n=3 → 6.4%, n=5 → 1.0% at equal FAR/MAR). Stop.

### 6d. LFP transfer on the BattGP field set (Zenodo 13715694, CC BY-NC 4.0, 1.7 GB)
1. Download into `data/battgp_lfp/` (ask me first). Reproduce the paper's per-cell fault probabilities
   for 2–3 systems (e.g. 3, 6, 9) as a correctness check. Reconcile 28 vs 29 systems and the sampling
   rate from the README.
2. KEY EXPERIMENT – pack-only ablation: build pack voltage = Σ cell voltages and drop per-cell data. Fit
   the pack-level OCV-free resistance GP (the SHS-realistic setting). Question: does the pack-level signal
   detect, and how many days later, the cell faults that the per-cell weakest-link method finds? Treat the
   per-cell method as pseudo-labels. Report detection rate, delay distribution and false flags.
3. Apply Phase 4 degradations (sampling interval, no temperature, history length, rounding, packet loss)
   to the pack-only LFP model. Deliver `results/minimum_data_table_lfp.csv`.
4. Lead-acid→LFP: apply Bboxx-trained models (6b) to BattGP features after per-system normalisation;
   report rank agreement only (no labels). Stop.

### 6e. RWTH home-storage field data (Zenodo 12091223, CC BY 4.0, ~21 GB – ask before downloading)
1. Identify the "Medium_LFP" systems from Metadata_and_Code.zip; download only those.
2. Fit the OCV-free pack resistance GP; compare DCR growth with Figgener et al. (LFP +0.8–1.8 pp/yr) as an
   external validity check. Quantify temperature confounding (housing T). Stop.

### 6f. (Optional) Temperature and DoD priors from Sandia LFP cells
Verify the Battery Archive licence first. Fit resistance/capacity-fade vs temperature (15/25/35 °C) and
DoD window. Export priors for the stress-factor features. Stop.

### 6g. Write-up and operator pack
1. Update REPORT.md with the cost-based results, benchmark table and pack-only LFP finding. State plainly
   whether pack-level telemetry is sufficient for LFP.
2. Create `docs/operator_data_request.md` (telemetry incl. per-cell V if available, BMS flags, repair
   records with {battery/other/no fault} outcome, trip cost by distance band, battery price, PAYGo
   instalment and outage repayment loss, churn, fleet size by age) and `docs/shadow_mode_protocol.md`
   (2–3 months of logged-not-acted alerts before any dispatch).
3. Update README, METHODOLOGY and LOG; commit.
```

## Conclusion

The SunFix pitch rests on a gap that the literature confirms but has not filled. GP-resistance methods run on LFP field data, but no one has shown they predict failure 30 or more days ahead, and no public LFP data from African SHS exists to test it. That makes the Zambia pilot data, and its labels, more valuable than any algorithm choice. The decisive unknown is whether **pack-level telemetry**, which most SHS likely send, can see the **single-cell faults** that seem to characterise LFP. Phase 6d can answer this cheaply with public data before any operator is approached. Commercially, the arithmetic recasts the product: at field base rates even a good classifier sends most trucks to healthy batteries. The thing to sell is therefore a **decision system** (calibrated risk, a remote confirmation step, an alert budget sized to technician capacity) priced against operator-supplied trip and outage costs, not a model accuracy figure.

## Sources

**Papers and preprints**
- Aitio & Howey 2021, *Joule* — https://arxiv.org/pdf/2107.13856 ; data https://intelligent-earth.ox.ac.uk/node/3976066 ; https://ora.ox.ac.uk/objects/uuid:e41d3d4c-f74e-4d76-81fd-0caa77ec6cec ; news https://www.ox.ac.uk/news/2021-12-15-new-approach-predicting-battery-failure-could-help-maintain-electricity-millions
- Schaeffer et al. 2024, *Cell Rep. Phys. Sci.* — https://arxiv.org/html/2406.19015v3 ; https://dspace.mit.edu/handle/1721.1/157659 ; code https://github.com/JoachimSchaeffer/BattGP
- Figgener et al. 2024, *Nature Energy* — https://ideas.repec.org/a/nat/natene/v9y2024i11d10.1038_s41560-024-01620-9.html ; https://www.ess-news.com/2024/09/18/most-home-battery-storage-systems-meet-warranty-claims-new-research-finds/
- Figgener et al. 2024 preprint — https://arxiv.org/pdf/2411.08025
- Cornejo et al. 2026 preprint — https://arxiv.org/pdf/2609.04487 ; https://github.com/martincornejo/BatteryRecursiveGPs.jl
- Aitio, Jöst, Sauer & Howey — https://arxiv.org/pdf/2304.13666
- Zhou, Aitio & Howey — https://arxiv.org/abs/2407.06639v4
- Zhou & Howey — https://arxiv.org/pdf/2211.05697
- Perriment et al. 2023 — https://arxiv.org/abs/2311.14098 ; Howey slides https://www.energy.ox.ac.uk/wp-content/uploads/2024/09/Howey_Oxford_Energy_day24_shortened.pdf
- Wang et al. 2023 — https://pure.bit.edu.cn:443/en/publications/large-scale-field-data-based-battery-aging-prediction-driven-by-s
- Nakano et al. 2024 — https://doaj.org/article/353ccf4394314ef491143dcf48081ef2
- DyAD 2023 — https://doaj.org/article/f727a539d2004bba9fce5332a35a3494
- EVBattery 2022 — https://huggingface.co/papers/2201.12358
- Voronov et al. 2020 — https://www.papers.phmsociety.org/index.php/ijphm/article/view/2608
- Xue et al. 2025 — https://arxiv.org/html/2503.13558v7
- Zhao et al. 2025 — https://arxiv.org/pdf/2505.05364
- Chen 2013 — https://papers.phmsociety.org/index.php/phmconf/article/view/2249 ; conformal GP (MDPI) https://www.mdpi.com/1999-4893/19/7/535
- Sulzer et al. 2021 (summary) — https://eng.ox.ac.uk/news/predicting-battery-lifetime-from-field-data
- Unverified leads — https://pure.bit.edu.cn/en/publications/state-of-health-estimation-for-lifeposub4subbattery-system-on-rea/fingerprints ; https://research.buaa.edu.cn/en/publications/state-of-charge-and-capacity-estimation-for-mwh-scale-lifeposub4s/ ; https://www.papers.phmsociety.org/index.php/ijphm/article/view/3115
- Review of battery data / large models 2026 — https://arxiv.org/pdf/2608.26111 ; BatteryML https://arxiv.org/pdf/2310.14714

**Datasets**
- BattGP LFP — https://zenodo.org/records/13715694
- RWTH home storage — https://zenodo.org/records/12091223
- Sandia — https://sandia.gov/research/publications/details/degradation-of-commercial-lithium-ion-cells-as-a-function-of-chemistry-and-2020-01-09 ; https://www.sandia.gov/ess/tools-resources/rd-data-repository ; https://www.batteryarchive.org/ [U]
- HUST — https://data.mendeley.com/datasets/compare/nsc7hnsg4s [S]
- MATR — https://data.matr.io/1/ [U] ; https://osbad.readthedocs.io/en/latest/ds_01_severson_dataset.html
- Turku — https://pmc.ncbi.nlm.nih.gov/articles/PMC12175228/
- CALCE — https://web.calce.umd.edu/batteries/data/
- Empa — https://zenodo.org/records/15481956
- Dataset list — https://github.com/lappemic/open-source-battery-data

**False positives and costs**
- Koops 2018 — https://www.bauhaus-luftfahrt.net/fileadmin/user_upload/Publikationen/2018_Koops_Business_Case_Analysis_Predictive_Maintenance_PHME.pdf ; https://papers.phmsociety.org/index.php/phme/article/download/460/phmec_18_460
- Elkan 2001 — https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations
- Niculescu-Mizil & Caruana 2005 — https://mlanthology.org/icml/2005/niculescumizil2005icml-predicting
- Adnan, Izadi & Chen 2011 — https://skoge.folk.ntnu.no/prost/proceedings/acc11/data/papers/1262.pdf
- Conformal anomaly detection — https://arxiv.org/pdf/2605.13642
- Kundu & Ramdas 2022 — https://ideas.repec.org/a/inm/ormsom/v24y2022i3p1329-1348.html
- SolarAid 2024 — https://solar-aid.org/news/launching-the-state-of-repair-report/ ; Lighting Global 2014 — https://www.lightingglobal.org/new-market-intelligence-on-after-sales-service-and-product-warranty/ ; Hystra 2019 — https://www.hystra.com/our-insights/pricing-quality-cost-drivers-and-value-add-in-the-off-grid-solar-sector-2019
- Not yet read: Power Africa PAYGO Guide https://uatweb.usaid.gov/sites/default/files/2022-05/Power-Africa-PAYGO-Guide-for-Off-grid-Energy-Companies.pdf ; GET.invest Zambia SHS https://www.get-invest.eu/wp-content/uploads/2020/11/GETinvest-Market-Insights_ZMB_SHS_-CS-Kazang_2019.pdf

**Market statistics**
- IEA GEVO 2026 — https://iea.org/reports/global-ev-outlook-2026/electric-vehicle-batteries
- IEA GEVO 2024 (secondary) — https://www.batterytechonline.com/ev-batteries/iea-report-ev-battery-prices-drop-lfp-surges-sodium-ion-on-horizon
- Adamas 2025 — https://www.adamasintel.com/lfp-now-commands-40-of-the-global-ev-battery-market/
- SNE Research 2026 — https://sneresearch.com/en/insight/release_view/681/page/0
- Benchmark 2026 — https://source.benchmarkminerals.com/article/global-lithium-ion-battery-demand-rose-29-in-2025
- CGD 2026 — https://www.cgdev.org/sites/default/files/lead-safe-off-grid-electrification-understanding-issues.pdf
- GOGLA 2019 — https://gogla.org/blog/how-lithium-batteries-can-help-advance-a-solar-powered-world/
- IEA PVPS Task 18 2025 — https://iea-pvps.org/wp-content/uploads/2025/10/TASK18-Chinese-Li-Ion-Battery-Systems-in-Off-Grid-Applications-2025.pdf
- Sun King 40Z — https://www.engineeringforchange.org/solutions/product/sun-king-home-40z/ ; Bboxx bPower — https://bboxx.com/products/bpower80-to-240-range
- SolarQuarter 2026 — https://solarquarter.com/2026/06/09/gogla-members-achieve-record-10-million-off-grid-solar-kit-sales-in-2025-expanding-energy-access-worldwide-report/
- IGC Zambia 2025 — https://www.theigc.org/sites/default/files/2025-07/Hardie-et-al-Policy-note-July-2025.pdf
