# Deep research prompt: LFP battery failure prediction from field data (SunFix)

## Context
SunFix is a software-only, manufacturer-neutral layer that predicts battery failure in off-grid solar home systems (SHS) in sub-Saharan Africa. It uses telemetry the systems already send: timestamp, current, voltage and temperature. Operators can then replace batteries before households lose power. Pilot market: Zambia.

Technical basis: **Aitio & Howey (2021, Joule)** fitted a Gaussian-process (GP) internal-resistance model to 1,027 Bboxx **lead-acid** batteries in the field. They reached 73% balanced accuracy for end-of-life prediction 8 weeks ahead and 82% at failure.

The SHS market is moving to **LFP (LiFePO4)**: RDG, Sun King, d.light and Ignite all ship it. LFP has a flat open-circuit-voltage curve, so voltage-based methods struggle.

**Schaeffer et al. (2024, Cell Reports Physical Science)** extended the GP-resistance approach to LFP field data, with the BattGP code and a Zenodo dataset of 28 returned 24 V packs.

**Why false positives matter.** In SunFix's business model each false alarm sends a technician on a costly rural trip. Reducing false positives at a useful warning horizon (≥30 days) is therefore the commercial priority.

## Research questions

### 1. Literature on LFP failure and health prediction from field or realistic data
- Find peer-reviewed papers and credible preprints from 2018 to today that predict state of health (SOH), remaining useful life, end of life, or failure/faults for **LFP batteries** from **field or operational data**, or from lab data under realistic, non-uniform conditions. Prioritise:
  - work that reproduces, extends or cites Aitio & Howey (2021)
  - work similar to Schaeffer et al. (2024)
- Separate the papers into: (a) field data from SHS, mini-grids, off-grid, telecom or stationary storage (BESS); (b) field data from EVs, e-buses, e-bikes or other light vehicles; (c) lab cycling data.
- For each paper give:
  - full citation, DOI or link
  - chemistry and application
  - dataset size (cells or systems, duration)
  - signals used
  - method
  - prediction target and horizon
  - headline metrics
  - whether code and data are public
  - one line on its relevance to SunFix
- Say explicitly whether anyone has tested the Aitio & Howey method on LFP, and what has been found about how well it transfers.

### 2. Datasets we can actually download
List every public LFP dataset usable for health or failure prediction. Known starting points to verify and extend:
- Severson et al. 2019 (A123 LFP/graphite)
- Attia et al. 2020
- the BattGP Zenodo field set
- HUST LFP data
- CALCE LFP cells
- Sandia (SNL) cycling data
- Battery Archive
- any field data from storage, telecom or vehicles

For each dataset give:
- name, and lab or field
- chemistry and form factor
- number of cells or systems
- signals and sampling rate
- labels available (EOL, failure, capacity fade)
- size in GB
- licence (commercial use allowed?)
- the direct download URL, verified to resolve
- how close it is to SHS conditions (heat, partial cycling, irregular charging)

Rank the top 5 for SunFix.

### 3. Algorithms worth benchmarking against the GP baseline
Find methods that have been applied to battery field data or similar predictive-maintenance problems, with evidence:
- gradient boosting
- survival analysis (Cox, random survival forests, DeepSurv)
- anomaly and change-point detection
- recurrent and transformer sequence models
- physics-informed or hybrid models
- transfer learning across chemistries and domain adaptation
- hierarchical Bayesian models
- conformal prediction

For each, give strengths, weaknesses and data needs, and recommend which 3–4 to benchmark, with justification. Note which ones handle censored data (batteries still alive), non-uniform sampling and sparse telemetry.

### 4. Reducing false positives
What does the literature on battery prognostics and industrial predictive maintenance recommend? Cover:
- cost-sensitive learning and decision-theoretic thresholds, i.e. choosing the alert threshold from the cost of a false alarm versus a missed failure
- probability calibration
- precision-at-k / alert budgets, i.e. capping alerts at what technicians can handle
- conformal or uncertainty-gated alerts
- persistence or hysteresis rules, e.g. the alert must hold for N days
- two-stage confirmation, e.g. a remote diagnostic before dispatch
- fleet-relative baselines

Find quantitative evidence where possible. Then propose an evaluation design for SunFix that optimises **expected cost per 1,000 systems per month**. Name the operational metrics:
- false dispatches per 1,000 systems
- missed failures
- lead-time distribution
- the cost model

List which inputs we need from operators: cost per technician trip, battery price, outage cost, and anything else required.

### 5. Market statistics: LFP versus other lithium-ion versus lead-acid
Answer with the most recent, sourced numbers:
- (a) **Global** lithium-ion battery demand and production by chemistry: LFP versus NMC/NCA/other (IEA Global EV Outlook, BNEF, Benchmark, etc.), split by EV and stationary storage where available. Include the trend over 2020 to the latest year.
- (b) **Off-grid solar / SHS:** what share of products sold in sub-Saharan Africa use LFP versus other lithium-ion versus lead-acid versus NiMH? Try GOGLA, Lighting Global / VeraSol product data (the VeraSol database lists battery chemistry for certified products, so a count or estimate is possible), Efficiency for Access, the ESMAP Off-Grid Solar Market Trends Report, and operator product sheets.
- (c) **Zambia** specifically if available.

Distinguish clearly between sales-weighted, product-count and capacity-weighted figures. Where data are missing, say so and give the best defensible proxy with its limits.

## Output format
1. **Executive summary** (≤10 bullets): what is known, what is not, and what SunFix should do next.
2. **Literature table** (Q1).
3. **Dataset table** (Q2) with a top-5 ranking and verified links.
4. **Algorithm shortlist** (Q3) with recommended benchmarks.
5. **False-positive strategy** (Q4) with an evaluation design and the operator data needed.
6. **Chemistry statistics** (Q5) with source tables.
7. **Recommended next experiments:** concrete steps that extend the existing Claude Code pipeline (BattGP on the Bboxx data), ready to paste in as "Phase 6".
8. **Sources:** full list with links.

## Rules
- Never invent papers, datasets, numbers or links. If something can't be verified, mark it "unverified" or leave it out.
- Prefer primary sources: papers, official dataset pages, IEA, GOGLA. Note the publication date for every statistic.
- Give confidence levels for key claims, and flag contradictions between sources.
- British English. Concise, tables where useful, no filler.
