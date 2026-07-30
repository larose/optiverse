VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: build
build:
	$(PYTHON) -m build

.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
#	find . -type d -name __pycache__ -exec rm -rf {} +
#	find . -type f -name "*.pyc" -delete

.PHONY: init
init:
	python3 -m venv $(VENV)
	$(PIP) install -e .[dev,agent]

.PHONY: format
format:
	$(VENV)/bin/black .

.PHONY: publish
publish: build
	$(PYTHON) -m twine upload dist/*

.PHONY: publish.test
publish.test: build
	$(PYTHON) -m twine upload --repository testpypi dist/*

.PHONY: run.tsp
run.tsp:
	$(PYTHON) -m examples.tsp.main

.PHONY: run.integer_compression
run.integer_compression: examples/integer_compression/harness/ts.txt
	$(PYTHON) examples/integer_compression/main.py

examples/integer_compression/harness/ts.txt:
	$(PYTHON) examples/integer_compression/data_generator.py

.PHONY: test
test: test.format test.types

.PHONY: test.format
test.format:
	$(VENV)/bin/black --check .

.PHONY: test.types
test.types:
	$(VENV)/bin/pyright
