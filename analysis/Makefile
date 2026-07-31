VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# The parent package goes in editable and first, so `optiverse` resolves to this
# checkout rather than to the release on PyPI. Everything this subproject needs
# beyond that — matplotlib — is declared in its own pyproject and stays here.
.PHONY: init
init:
	python3 -m venv $(VENV)
	$(PIP) install -e .. -e '.[dev]'

# DIRECTORY is a run directory, the same name the examples use to point at one.
.PHONY: analyze
analyze:
	$(PYTHON) -m optiverse_analysis $(DIRECTORY)

.PHONY: clean
clean:
	rm -rf *.egg-info/

.PHONY: format
format:
	$(VENV)/bin/black .

.PHONY: test
test: test.format test.types test.unit

.PHONY: test.format
test.format:
	$(VENV)/bin/black --check .

.PHONY: test.types
test.types:
	$(VENV)/bin/pyright

.PHONY: test.unit
test.unit:
	$(PYTHON) -m unittest discover -s . -p "*_test.py" -v
