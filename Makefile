SHELL := /bin/bash
PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
PORT ?= 8010
API ?= http://localhost:$(PORT)
KIND_CLUSTER ?= traintrack

.DEFAULT_GOAL := help

.PHONY: help install up down migrate api worker snapshot seed test test-integration lint fmt kind-up kind-down train bench results

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install:  ## Create the venv and install backend dependencies
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip wheel setuptools
	$(PIP) install -e ".[dev]"

up:  ## Start Postgres, Redis, RabbitMQ, Prometheus, Grafana
	docker compose up -d --wait

down:  ## Stop local dependencies (volumes are preserved)
	docker compose down

migrate:  ## Apply Alembic migrations
	$(PYTHON) -m alembic upgrade head

api:  ## Run the Flask API (PORT=8010 by default)
	$(PYTHON) -m traintrack.api

worker:  ## Run one scheduler worker
	$(PYTHON) -m traintrack.worker

seed:  ## Create a developer API key and print it
	$(PYTHON) scripts/seed_dev.py

snapshot:  ## Build (or reuse) the MNIST dataset snapshot
	$(PYTHON) -m traintrack.pipeline.preprocess --version mnist-v1

test:  ## Run the backend test suite with coverage
	$(PYTHON) -m pytest -q --cov=traintrack --cov-report=term --cov-fail-under=80

test-integration:  ## Run the local end-to-end integration suite
	$(PYTHON) -m pytest -q -m integration

lint:  ## Ruff + mypy
	$(PYTHON) -m ruff check traintrack tests scripts
	$(PYTHON) -m mypy traintrack

fmt:  ## Auto-fix lint issues
	$(PYTHON) -m ruff check --fix traintrack tests scripts

kind-up:  ## Create the kind cluster and deploy TrainTrack
	./scripts/kind-up.sh

kind-down:  ## Delete the kind cluster
	kind delete cluster --name $(KIND_CLUSTER)

train:  ## Run a local training job against the configured snapshot
	$(PYTHON) -m traintrack.training.train --help

bench:  ## Run the submission/throughput benchmark
	$(PYTHON) scripts/bench_submit.py

results:  ## Regenerate docs/results/*.json from the live registry
	$(PYTHON) scripts/export_results.py
