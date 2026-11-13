# Contributing to TrainTrack

Thanks for helping improve TrainTrack. This guide covers the local loop and the rules that keep the
project's measurements honest.

## Setup

```bash
cp .env.example .env
make install          # Python 3.11 venv with dev dependencies
make up               # Postgres, Redis, RabbitMQ, Prometheus, Grafana
make migrate          # Alembic schema
```

Run the API and a worker in separate terminals with `make api` and `make worker`. The API bootstraps
the development key configured in `.env`.

For the dashboard:

```bash
cd frontend
npm ci
npm run dev
```

## Quality gates

```bash
make lint             # ruff + mypy
make test             # pytest with coverage, fails below 80%
cd frontend && npm run lint && npm run build && npm test
```

Install the hooks once with `pre-commit install`. The backend integration suite requires the compose
services to be healthy and uses separate test database/queue names via `TEST_DATABASE_URL`,
`TEST_REDIS_URL`, and `TRAINTRACK_QUEUE_PREFIX`; stop stray development workers before running the
benchmark because it deliberately fails if another consumer is attached.

## Pull requests

- Keep changes consistent with the existing module boundaries: API blueprints in `traintrack/api`,
  domain services in `traintrack/services`, executors in `traintrack/executors`, pipeline stages in
  `traintrack/pipeline`, and training code in `traintrack/training`.
- Add tests for behavior changes. State-machine and idempotency changes need explicit tests.
- Run `alembic revision --autogenerate -m "..."` for model changes and commit the migration; never
  edit an applied migration.
- Do not hand-edit numbers in the README. Metrics come from `docs/results/*.json`; regenerate with
  `make results` and `python scripts/update_readme_results.py`.
- Document operational changes (env vars, RBAC, ports, storage) in `.env.example` and the README.
- Never commit secrets or real credentials. The values in `deploy/k8s/secret.yaml` are dev-only.

## Commit style

Use short imperative subjects, for example `feat: add artifact lineage endpoint` or
`fix: recover orphaned runs after worker loss`. Keep unrelated changes in separate commits.

## Reporting problems

When opening an issue, include the run id, relevant status transitions
(`GET /api/runs/<id>`), worker logs, and whether the executor was local or Kubernetes. That triplet
usually identifies whether the failure is in scheduling, execution, or persistence.
