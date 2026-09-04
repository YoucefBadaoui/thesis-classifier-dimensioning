PY ?= python

.PHONY: test tables figures verify all

test:
	$(PY) -m pytest tests/ -q

tables:
	$(PY) scripts/regenerate_analytical_results.py

figures: tables
	$(PY) scripts/regenerate_all_figures.py

verify:
	$(PY) scripts/checks/verify_core_math.py
	$(PY) scripts/checks/verify_jensen_convexity.py
	$(PY) scripts/checks/verify_rebinning_flip.py
	$(PY) scripts/checks/validate_pipeline.py

all: test verify tables figures
