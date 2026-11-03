# SunFix: standing instructions for Claude Code

## Project summary
SunFix predicts battery failure in off-grid solar home systems from telemetry the systems already send. This repo applies the open-source BattGP library (Schaeffer et al. 2024) to the Bboxx lead-acid field dataset of Aitio & Howey (2021) (1,027 batteries, CC BY-NC 4.0, research code only). Two questions: (1) reproduce 82% / 73% balanced accuracy at failure / 8 weeks before; (2) what is the minimum telemetry a manufacturer must ship for a useful 30-day warning. Original brief: `claude-code-prompt-battgp-bboxx.md`. Later phases: `PHASE6_PROMPT.md`.
Package is `sunfix/` (not `src/`), because BattGP imports a top-level `src`.

## Rules
- Never modify or delete the original zips or `meta_data.csv`. Write everything new under `data/`, `src/`, `notebooks/`, `results/` and `docs/`.
- No look-ahead leakage: features at time *t* may only use data up to *t*. Split by battery, not by row.
- Fix random seeds and make every result reproducible with one command (`make all` or `python -m src.run_all`).
- Commit after each phase with a clear message. Follow the session protocol: no session or phase ends without updating `LOG.md` and `README.md`, plus `METHODOLOGY.md` when methods changed.
- If something in BattGP doesn't fit our data, explain the problem and the options before making a large workaround.
- Tell me before any step that will take more than 30 minutes or use more than about 16 GB of RAM.
- Explain results in plain language. I'm not a battery engineer; I need to tell an operator, an investor and a ONE Earth reviewer what this shows.

## Session protocol



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
