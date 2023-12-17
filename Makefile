.DEFAULT_GOAL:=help
SHELL:=/bin/bash

.EXPORT_ALL_VARIABLES:
PYTHONPATH := source/${PYTHONPATH}

help:
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*?##/ { printf "  \033[36m%-10s\033[0m -%s\n", $$1, $$2 }' $(MAKEFILE_LIST)

lint:
	ruff format --diff --quiet --check .	
	ruff check . --fix --show-fixes

format:
	ruff format .
	ruff check . --fix

mypy_check:
	mypy --strict .

run_tests: 
	IS_LOCAL_RUN=1 pytest -svvv tests/

pre_push_test: lint mypy_check run_tests
