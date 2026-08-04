VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# The run to draw: `make run RUN=../tmp/20260804_110031`.
RUN =

.PHONY: init
init:
	python3 -m venv $(VENV)
# The parent from this checkout rather than from PyPI, so the visualizer reads
# the graph the code in this tree writes.
	$(PIP) install -e ..
	$(PIP) install -e .[dev]

.PHONY: run
run:
	$(PYTHON) -m optiverse_viz $(RUN)

.PHONY: format
format:
	$(VENV)/bin/black .

.PHONY: test
test: test.format test.types

.PHONY: test.format
test.format:
	$(VENV)/bin/black --check .

.PHONY: test.types
test.types:
	$(VENV)/bin/pyright
