VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

# The Go in this repo is example harnesses and one seed codebase. Black never saw
# any of it, which is how the seed drifted out of gofmt shape unnoticed.
GO_SOURCES = $(shell find examples -name '*.go')

# Only one example needs Go, so a checkout without it should still be able to
# format and test the Python.
SKIP_WITHOUT_GOFMT = command -v gofmt > /dev/null || \
	{ echo "gofmt not installed; skipping the Go sources"; exit 0; };

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
format: format.python format.go

.PHONY: format.python
format.python:
	$(VENV)/bin/black .

.PHONY: format.go
format.go:
	@$(SKIP_WITHOUT_GOFMT) \
	gofmt -w $(GO_SOURCES)

.PHONY: publish
publish: build
	$(PYTHON) -m twine upload dist/*

.PHONY: publish.test
publish.test: build
	$(PYTHON) -m twine upload --repository testpypi dist/*

.PHONY: run.tsp
run.tsp:
	$(PYTHON) -m examples.tsp.optimize

.PHONY: run.integer_compression
run.integer_compression: examples/integer_compression/harness/ts.bin
	$(PYTHON) -m examples.integer_compression.optimize

examples/integer_compression/harness/ts.bin:
	$(PYTHON) examples/integer_compression/harness/dataset.py

.PHONY: test
test: test.format test.types test.unit

.PHONY: test.format
test.format: test.format.python test.format.go

.PHONY: test.format.python
test.format.python:
	$(VENV)/bin/black --check .

.PHONY: test.format.go
test.format.go:
	@$(SKIP_WITHOUT_GOFMT) \
	unformatted="$$(gofmt -l $(GO_SOURCES))"; \
	if [ -n "$$unformatted" ]; then \
		echo "not gofmt-clean:"; \
		echo "$$unformatted"; \
		echo "run: make format.go"; \
		exit 1; \
	fi

.PHONY: test.types
test.types:
	$(VENV)/bin/pyright

.PHONY: test.unit
test.unit:
	$(PYTHON) -m unittest discover -s . -p "*_test.py" -v
