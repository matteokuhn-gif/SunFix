# Reproduce everything from the raw zips (about 35 min on a 10-core laptop). Research code, CC BY-NC data.
PY = .venv/bin/python
.PHONY: all phase1 phase2 phase3 pdf
all: phase1 phase2 phase3 pdf
phase1:
	$(PY) -m sunfix.summarise && $(PY) -m sunfix.cache_all && $(PY) -m sunfix.eda
phase2:
	$(PY) -m sunfix.fit_hyper ocv && $(PY) -m sunfix.fit_hyper nocv && $(PY) -m sunfix.phase2
phase3:
	$(PY) -m sunfix.phase3_features && $(PY) -m sunfix.classify && $(PY) -m sunfix.phase3_figure
pdf:
	$(PY) -m sunfix.make_pdf
