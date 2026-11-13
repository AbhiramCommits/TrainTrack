# TrainTrack

TrainTrack is a self-hostable ML experiment, data-pipeline, and artifact platform. Teams submit
training and evaluation jobs, the platform schedules them onto Kubernetes (with a local subprocess
executor for development), tracks every run durably in PostgreSQL, streams live logs, stores
content-addressed artifacts with lineage back to the producing run and dataset snapshot, and exports
Prometheus metrics for dashboards and alerts.

It solves the operational problem that appears once training moves beyond one laptop notebook:
which code, data snapshot, hyperparameters, and device produced a checkpoint; which runs are queued
or stuck; why a job failed; and how much of a cluster is idle. TrainTrack owns the run state
machine, retries, dead letters, artifact dedupe, and lineage so the answers survive restarts and are
queryable through an API and dashboard.

## 1. What it does

- **Schedules** real training jobs on Kubernetes Jobs or a local subprocess executor, with retries,
  exponential backoff, orphan recovery, cancellation, and a dead-letter queue.
- **Tracks** every run in a PostgreSQL registry with an append-only status-transition audit,
  keyset-paginated listing, queue positions, idempotent submission, and per-attempt history.
- **Streams** stdout/stderr through Redis Streams to a Server-Sent Events endpoint and the dashboard.
- **Versions data** with deterministic MNIST snapshots, content hashes, manifests, and a validation
  gate that refuses unvalidated or hash-mismatched snapshots.
- **Trains and evaluates** a real PyTorch CNN (CPU/MPS; CUDA path implemented) and records every
  metric from the actual run.
- **Stores artifacts** by SHA-256 in a content-addressed store, deduping identical writes and
  recording lineage (`parent_sha`) from evaluation reports to checkpoints to runs to snapshots.
- **Observes** the system with Prometheus counters, histograms, gauges, alert rules, and a
  provisioned Grafana dashboard.
- **Visualizes** runs, metrics, live logs, comparison curves, lineage, GPU/device samples, and queue
  health in a React + TypeScript + Redux dashboard.

## 2. Architecture

```mermaid
flowchart LR
    UI["React 18 + TypeScript\nRedux Toolkit / RTK Query"] -->|REST + SSE| API["Flask API\napp factory + blueprints"]
    API --> PG[("PostgreSQL 16\nruns, transitions,\nmetrics, snapshots,\nartifacts")]
    API --> REDIS[("Redis 7\nleaderboard cache,\nrate limits,\nlog streams,\nresource samples")]
    API -->|publish run| EX{{"RabbitMQ 3.13\nexchange: traintrack"}}
    EX -->|runs.submitted| W["Worker scheduler\nprefetch=1, SELECT FOR UPDATE\nSKIP LOCKED claim"]
    W -.->|TTL retry + backoff| EX
    EX -->|runs.dead| DLQ[("DLQ\nruns.dead")]
    W -->|TRAINTRACK_EXECUTOR=local| LOCAL["LocalExecutor\nsubprocess + timeout"]
    W -->|TRAINTRACK_EXECUTOR=kubernetes| K8S["Kubernetes executor\nbatch/v1 Job\nnvidia.com/gpu when requested"]
    K8S --> POD["Job Pod\n(CPU/MPS or GPU)"]
    LOCAL --> PY["PyTorch train/eval\nreal CNN + metrics"]
    POD --> PY
    PY -->|POST metrics + resources| API
    PY -->|content-addressed writes| CAS[("Artifact store\nsha256 -> artifacts/aa/hash")]
    API -->|/metrics scrape| PROM["Prometheus"] --> GRAF["Grafana"]
```

Workers claim with `SELECT ... FOR UPDATE SKIP LOCKED`, so running several workers is safe. A Redis
execution lock plus a heartbeat prevents two workers from recovering the same execution; if a worker
dies, the lock expires, the redelivered message moves the run through `FAILED -> QUEUED`, and a
delayed retry gives it back to the scheduler.

### Run state machine

```text
                    +-----------------------------+
                    |            FAILED           |
                    |  (retry when attempts left) |
                    +--------------+--------------+
                           ^       |
                           |       | FAILED -> QUEUED (backoff + TTL retry)
                           |       v
QUEUED ------------> SCHEDULED ------------> RUNNING ------------> SUCCEEDED
  |                      |                      |
  |                      |                      +------------------> FAILED
  |                      |                      +------------------> CANCELED
  +----------------------+-----------------------------------------> CANCELED
```

All transitions are validated by `traintrack/models/state.py`; illegal transitions raise and are unit
tested. Every change is appended to `status_transitions` with actor and reason.

### Lineage model

```text
eval_report artifact --parent_sha--> checkpoint artifact --parent_sha--> earlier checkpoint
        |                                  |
        +--> run (hyperparams, git_sha) ---+--> dataset_snapshot (version, content_hash, manifest)
```

`GET /api/artifacts/<sha>/lineage` walks `parent_sha` and joins each artifact to its producing run
and dataset snapshot. The content-addressed store uses deterministic checkpoint serialization, so
identical checkpoint contents hash identically and are stored once.

## 3. How to run

### Prerequisites

- Python 3.11 and `make`
- Docker Desktop with Docker Compose
- Node.js 20+ and npm
- Optional for the Kubernetes path: `kind`, `kubectl`, and enough Docker memory (~8 GB)

Ports were chosen to avoid clashing with other local services on this machine: PostgreSQL `55432`,
Redis `56379`, RabbitMQ `5672/15672`, API `8010`, frontend `5273`, Prometheus `9090`, Grafana `3000`.
Everything is configurable in `.env`. The API default is `8010`; set `TRAINTRACK_API_PORT` and
`VITE_PROXY_TARGET` if you prefer another port (the original brief's `8000` was occupied here).

### 1. Start dependencies and install

```bash
cp .env.example .env
make install
make up          # Postgres, Redis, RabbitMQ, Prometheus, Grafana; waits for health
make migrate     # Alembic upgrade head
```

### 2. Start the API and a worker

In separate terminals:

```bash
make api         # http://localhost:8010/healthz
make worker      # consumes runs.submitted with the configured executor
```

The API bootstraps the development key `tt_dev_local_key_do_not_use_in_prod` on first start. Use
`make seed` to create a named key and a demo experiment. For production, set a strong
`TRAINTRACK_DEV_API_KEY`/create managed keys and never ship the dev key.

### 3. Build the dataset snapshot

```bash
make snapshot    # downloads MNIST, writes deterministic shards, hashes, validates, registers
```

Output is `data/snapshots/mnist-v1/` with `manifest.json`, `validation.json`, and six `.npy` shards.
The content hash covers every shard and `class_names.json`.

### 4. Submit a run via curl

```bash
export TT_KEY=tt_dev_local_key_do_not_use_in_prod
export API=http://localhost:8010

SNAP=$(curl -sS "$API/api/snapshots" -H "X-API-Key: $TT_KEY" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["snapshots"][0]["id"])')
EXP=$(curl -sS -X POST "$API/api/experiments" -H "X-API-Key: $TT_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"name":"my-first-experiment","description":"curl demo"}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')

curl -sS -X POST "$API/api/runs" -H "X-API-Key: $TT_KEY" \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: demo-run-001' \
  -d "{\"experiment_id\":\"$EXP\",\"job_kind\":\"train\",
       \"hyperparams\":{\"lr\":0.001,\"batch_size\":128,\"epochs\":1,\"optimizer\":\"adam\"},
       \"dataset_snapshot_id\":\"$SNAP\",\"requested_gpus\":1}"
```

Poll `GET /api/runs/<id>`, read `GET /api/runs/<id>/metrics`, stream
`GET /api/runs/<id>/logs/stream?api_key=$TT_KEY`, and inspect
`GET /api/artifacts/<sha>/lineage`.

### 5. Run the dashboard

```bash
cd frontend
npm ci
npm run dev      # http://127.0.0.1:5273, Vite proxies /api to the Flask API
```

`VITE_API_KEY` and `VITE_PROXY_TARGET` in `frontend/.env.local` control the key and API target. The
dashboard polls runs and queue health, opens the SSE log stream, renders metric and device curves,
and compares runs.

### 6. Kubernetes path (kind)

```bash
make kind-up     # kind cluster, image build/load, migrations, snapshot registration, API + worker
kubectl -n traintrack port-forward svc/traintrack-api 18011:8010
python3 scripts/run_k8s_experiments.py --api-url http://localhost:18011
python3 scripts/export_k8s_results.py
```

`kind-up.sh` mounts `data/` and `artifacts/` into the node so Job pods and the API share the snapshot
and artifact store. The GPU manifest is validated with `kubectl apply --dry-run=server`. Tear down
with `make kind-down` and `make down`.

### 7. Tests, lint, benchmark, results

```bash
make test        # pytest with coverage, --cov-fail-under=80
make lint        # ruff + mypy
cd frontend && npm test && npm run lint && npm run build
make bench       # stop other workers first; writes docs/results/benchmark.json
make results     # regenerate measured result files (and the README results section)
```

The benchmark refuses to run if another worker is already consuming `runs.submitted`, then starts
1/2/4 worker processes and measures submission throughput, broker confirms, and drain time.

<!-- RESULTS:START -->

## 4. Results (measured)

Every figure in this section is rendered from `docs/results/*.json` by `scripts/update_readme_results.py`; those files are written by the build commands, not typed by hand. Files: [training](docs/results/training_runs.json), [Kubernetes](docs/results/k8s_runs.json), [benchmark](docs/results/benchmark.json), [reliability](docs/results/reliability.json), [tests](docs/results/test_results.json).

### 4.1 Hardware and dataset snapshot

All training runs below ran on the local machine described by the registry: `macOS-27.2.1-arm64-arm-64bit` (arm64), Python 3.11.14, PyTorch 2.5.1. CUDA available: **False**; MPS available: **True**. The devices actually used were: `mps`. No NVIDIA GPU was present. The `nvidia.com/gpu` request path is implemented and verified by rendered-manifest golden tests plus a Kubernetes server-side dry run (Section 4.5); no GPU benchmark is claimed.

| Snapshot | Content hash | Rows | Seed | Mean | Std |
|---|---|---|---|---|---|
| `mnist-v1` | `ec15555b070325cf62402d2c117c7327532b6c597a2bdf14251a28865250b2cb` | 70,000 | 42 | 0.130653 | 0.308102 |

| Split | Rows | Shape |
|---|---|---|
| train | 54,000 | `[54000, 1, 28, 28]` |
| val | 6,000 | `[6000, 1, 28, 28]` |
| test | 10,000 | `[10000, 1, 28, 28]` |

The snapshot was built with a fixed seed, normalized with the train-split mean/std, and registered only after the validation gate recomputed every file digest and the aggregate content hash.

### 4.2 Real training runs

5 training runs were submitted through the REST API, scheduled by the RabbitMQ worker, and actually trained with PyTorch. Wall clock is the registry's `finished_at - started_at`; train loop is the job-internal timer, including test evaluation.

| Run | Optimizer | LR | Batch | Epochs | Wall clock (s) | Train loop (s) | Val accuracy | Test accuracy | Device | Requested GPUs |
|---|---|---|---|---|---|---|---|---|---|---|
| `9b3a2bdf` | sgd | 0.01 | 64 | 2 | 39.13 | 29.91 | 98.45% | 98.51% | mps | 1 |
| `302871cc` | adam | 0.001 | 64 | 2 | 40.70 | 30.59 | 98.30% | 98.69% | mps | 1 |
| `fc1e38b9` | adam | 0.002 | 256 | 2 | 34.71 | 24.36 | 98.23% | 98.22% | mps | 1 |
| `483e480d` | adam | 0.0005 | 128 | 2 | 25.02 | 20.43 | 97.70% | 98.28% | mps | 1 |
| `2cb95fd1` | adamw | 0.0003 | 512 | 2 | 20.75 | 14.27 | 95.17% | 96.02% | mps | 1 |

Best validation run: `9b3a2bdf-0282-49e5-b167-5b244d79d3e0` with val accuracy **98.45%** and test accuracy **98.51%**. Its checkpoint digest is `20d740600b06eb4c81056a582134675ce73d0324b0b0d2400ec2e2910683da08`.

### 4.3 Best model evaluation report

A separate eval job loaded the best checkpoint, verified its dataset snapshot hash, and scored all 10,000 held-out test images.

| Metric | Value |
|---|---|
| Test accuracy | **98.51%** |
| Macro precision | 98.53% |
| Macro recall | 98.49% |
| Macro F1 | **98.50%** |
| Test cross-entropy loss | 0.046843 |
| Inference latency p50 | 0.015795 ms/sample |
| Inference latency p95 | 0.032448 ms/sample |
| Batch size | 512 |
| Evaluation wall clock | 3.57 s |
| Device | `mps` |

Per-class precision, recall, F1, support, and the full 10×10 confusion matrix are stored in the `eval_report` artifact and returned by the lineage endpoint.

### 4.4 Artifact store, dedupe, and lineage

| Artifact metric | Value |
|---|---|
| Unique artifacts | **21** |
| Artifact kinds | checkpoint: 10, eval_report: 6, train_summary: 5 |
| Stored bytes | 7.95 MiB |
| Attempted writes | 11.91 MiB |
| Deduped bytes saved | **3.96 MiB** |
| Dedupe ratio | **33.26%** of attempted bytes |
| Best-checkpoint lineage chain | 2 artifacts |
| Eval report points at best checkpoint | True |

Checkpoints use a deterministic `.npz` serialization (tensor bytes plus canonical JSON metadata), so writing the final-epoch checkpoint when it is also the best epoch is byte-identical and is deduped by the content-addressed store instead of stored twice. Lineage is queryable with `GET /api/artifacts/<sha>/lineage`: `artifact -> producing run -> hyperparams/git_sha -> dataset snapshot`, following `parent_sha` for chains.

### 4.5 Kubernetes executor on kind

Two runs were submitted through the API and executed as real `batch/v1` Jobs on a kind cluster running Kubernetes v1.37.0 (Debian GNU/Linux 13 (trixie), 8 CPUs, 8124776Ki memory, containerd://2.3.4). Scheduling latency is submit → pod `Running`; queue wait is submit → scheduler claim. The second run was queued behind the first because the cluster ran one worker, so its submit-to-running latency includes that queue wait.

| Run | Pod | Submit → pod Running (s) | Queue wait (s) | Pod execution (s) | End-to-end (s) | Val acc | Test acc |
|---|---|---|---|---|---|---|---|
| `e2829cb4` | `traintrack-run-e2829cb4-974a-4841-ac82-420ed9f03a7c-mwzkx` | 1.96 | 0.45 | 174.04 | **174.61** | 97.42% | 97.97% |
| `d2ee183e` | `traintrack-run-d2ee183e-5b64-4000-a948-69c4f5194acd-k9hj6` | 178.08 | 176.98 | 104.70 | **281.68** | 96.85% | 97.45% |

Observed submit → pod Running: min **1.96 s**, median **90.02 s**, max **178.08 s**. End-to-end wall time: min **174.61 s**, median **228.15 s**, max **281.68 s**.

The GPU request manifest (one `nvidia.com/gpu` limit plus GPU node selector and toleration) was validated with `kubectl apply --dry-run=server -f /tmp/traintrack-gpu-job.yaml`: **validated=True**. No GPU node existed, so the GPU path was not scheduled; the CPU path above is the real execution evidence.

Prometheus (the docker-compose instance, scraping `host.docker.internal:8010`) scraped the API `/metrics` endpoint (`up{job="traintrack-api"} = 1.0`). The Grafana dashboard `TrainTrack` is provisioned from `deploy/monitoring/dashboards/traintrack.json`; the cluster API exposes the same `/metrics` endpoint.

### 4.6 Submission benchmark

30 no-op runs per configuration, 6 concurrent submitters. Runs are queued first; the configured number of workers is then started and the backlog drain time is measured. No-op jobs isolate API, broker-confirm, and scheduler overhead from model training. Raw latency samples are in `docs/results/benchmark.json`. Throughput is wall-clock, so absolute values depend on concurrent host load.

| Workers | Submitted | Submit throughput (runs/s) | Submit p50 (ms) | Submit p95 (ms) | Submit p99 (ms) | Drain (s) | Drain throughput (runs/s) | Statuses |
|---|---|---|---|---|---|---|---|---|
| 1 | 30 | **76.23** | 65.84 | 134.71 | 140.75 | **2.15** | **13.95** | SUCCEEDED: 30 |
| 2 | 30 | **66.78** | 79.46 | 146.74 | 158.11 | **1.17** | **25.60** | SUCCEEDED: 30 |
| 4 | 30 | **74.18** | 78.99 | 98.25 | 107.05 | **0.65** | **46.43** | SUCCEEDED: 30 |

RabbitMQ publisher-confirm latency over 40 persistent publishes: p50 **0.95 ms**, p95 **2.30 ms**, p99 **3.75 ms**.

### 4.7 Retry and dead-letter behavior

A deliberately failing training job (snapshot path removed after registration) was submitted with `max_attempts=2` and observed through the real worker:

| Measurement | Value |
|---|---|
| Final status | `FAILED` |
| Attempts observed / configured | **2 / 2** |
| Retry transitions | 1 |
| Observed retry backoff | **2.071 s** |
| Messages in `runs.dead` | **1** |
| Failure reason | `process exited with 2` |

The same suite also exercises broker disconnect/reconnect (the worker reconnects and finishes a subsequent run), orphaned `RUNNING` run recovery after worker loss (the execution lock expires, the run is failed, requeued, and retried), and cancellation of a running job (the executor observes the cancel flag and the run ends `CANCELED`).

### 4.8 Test suite and coverage

| Suite | Tests | Passed | Failed | Coverage |
|---|---|---|---|---|
| Backend (`pytest`) | 86 | 86 | 0 | **85.81%** |
| Frontend (`vitest`) | 21 | 21 | 0 | — |
| **Total** | **107** | **107** | **0** | — |

Coverage is measured over `traintrack/` with `--cov-fail-under=80` enforced. Scope: traintrack/ (training entry points, preprocess, and k8s executor excluded). The integration test exercises API → RabbitMQ → worker → LocalExecutor → PyTorch metrics → artifact lineage against the real compose services.

<!-- RESULTS:END -->

## 5. API reference

All data endpoints require `X-API-Key` (or `Authorization: Bearer`). The SSE endpoint also accepts
`?api_key=` because browser `EventSource` cannot set headers.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Liveness |
| `GET` | `/readyz` | Postgres + Redis readiness |
| `GET` | `/metrics` | Prometheus exposition (no key) |
| `POST` | `/api/experiments` | Create experiment |
| `GET` | `/api/experiments` | List experiments with run counts |
| `GET` | `/api/experiments/<id>` | Experiment detail and runs |
| `GET` | `/api/experiments/<id>/leaderboard` | Best validation score per run, 30s Redis cache, invalidated on metric write |
| `POST` | `/api/runs` | Submit run; requires `Idempotency-Key`; replay returns 200 |
| `GET` | `/api/runs` | Filter by experiment/status/kind with keyset pagination |
| `GET` | `/api/runs/<id>` | Detail, queue position, full transition timeline |
| `POST` | `/api/runs/<id>/cancel` | Cancel queued/scheduled/running run |
| `POST` | `/api/runs/<id>/metrics` | Batch metric ingest (≤1000 points) |
| `GET` | `/api/runs/<id>/metrics` | Metric points, optional `names` filter |
| `POST` | `/api/runs/<id>/resources` | Batch device utilization/memory samples |
| `GET` | `/api/runs/<id>/resources` | Device samples for the utilization panel |
| `GET` | `/api/runs/<id>/logs?tail=N` | Recent log lines |
| `GET` | `/api/runs/<id>/logs/stream` | SSE replay then tail |
| `GET` | `/api/artifacts?run_id=&kind=` | Artifact listing |
| `GET` | `/api/artifacts/<sha>/lineage` | Artifact → run → hyperparams/git_sha → snapshot chain |
| `GET` | `/api/snapshots` | Validated snapshots for the picker |
| `GET` | `/api/snapshots/<id>` | Snapshot manifest detail |
| `GET` | `/api/queue/health` | Work/retry/DLQ depths from RabbitMQ |

Errors are JSON (`{"error","message"}`) with `400/401/404/409/422/429/503` as appropriate.

## 6. Configuration

Key environment variables (see `.env.example` for the full list):

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://traintrack:traintrack@localhost:55432/traintrack` | Run registry |
| `REDIS_URL` | `redis://localhost:56379/0` | Cache, rate limits, log/resource streams |
| `AMQP_URL` | `amqp://traintrack:traintrack@localhost:5672/%2F` | RabbitMQ |
| `TRAINTRACK_API_PORT` | `8010` | Flask port |
| `TRAINTRACK_API_URL` | `http://localhost:8010` | URL used by training jobs to post metrics |
| `TRAINTRACK_DEV_API_KEY` | dev-only value | Bootstrapped developer key |
| `TRAINTRACK_RATE_LIMIT` | `120` | Requests/minute per key |
| `TRAINTRACK_EXECUTOR` | `local` | `local` or `kubernetes` |
| `TRAINTRACK_MAX_ATTEMPTS` | `3` | Default retry budget |
| `TRAINTRACK_RETRY_BASE_SECONDS` / `_MAX_SECONDS` | `2` / `60` | Exponential backoff |
| `TRAINTRACK_EXEC_LOCK_TTL` | `30` | Seconds before a crashed worker's lock expires |
| `TRAINTRACK_ARTIFACT_ROOT` | `./artifacts` | Content-addressed store |
| `TRAINTRACK_DATA_ROOT` | `./data` | Raw data and snapshots |
| `TRAINTRACK_K8S_NAMESPACE` / `_IMAGE` | `traintrack` / `traintrack:dev` | Kubernetes executor target |
| `TRAINTRACK_K8S_CPU_REQUEST` / `_LIMIT` | `1` / `4` | Job resources |
| `TRAINTRACK_K8S_MEMORY_REQUEST` / `_LIMIT` | `1Gi` / `6Gi` | Job resources |
| `TRAINTRACK_K8S_GPU_NODE_SELECTOR_KEY` | `runpod.io/gpu` | GPU node selector key |
| `TRAINTRACK_LEADERBOARD_TTL` | `30` | Leaderboard cache seconds |
| `REDIS_LOG_MAXLEN` | `10000` | Capped log stream length |
| `TRAINTRACK_QUEUE_PREFIX` | empty | Queue-name prefix used to isolate test/dev stacks |

Kubernetes manifests also read `TRAINTRACK_K8S_DATABASE_URL`, `TRAINTRACK_K8S_REDIS_URL`, and
`TRAINTRACK_K8S_API_URL` so Job pods talk to in-cluster services.

## 7. Security notes

- API keys are stored as SHA-256 digests with an 8-character display prefix; verification uses
  `hmac.compare_digest`. Plaintext keys are never persisted.
- Per-key fixed-window rate limiting lives in Redis. Redis outages fail open so the control plane
  stays available; the trade-off is documented here rather than hidden.
- The Kubernetes worker ServiceAccount has a namespaced Role limited to `jobs`, `jobs/status`, `pods`,
  `pods/log`, and the verbs the executor actually uses. Job pods use a separate ServiceAccount with
  no API permissions.
- Credentials are supplied through environment variables and Kubernetes Secrets. The checked-in
  `deploy/k8s/secret.yaml` contains development-only values for kind and must be replaced in
  production (external secret manager, sealed secrets, or SOPS).
- The SSE endpoint accepts the API key in a query parameter only because `EventSource` cannot send
  custom headers; access logs should redact query strings.
- The dashboard reads its API key from build-time env (`VITE_API_KEY`). It is an operator console,
  not a public multi-user app; put it behind SSO/reverse-proxy auth before exposing it.
- Inputs are validated with pydantic and the ORM uses bound parameters. SQL identifiers are static.
- TLS termination, per-user accounts, and tenant isolation are intentionally out of scope here; see
  limitations.

## 8. Known limitations and next steps

- **No NVIDIA GPU was available.** Real training ran on Apple Silicon MPS. The CUDA device path,
  `nvidia.com/gpu` limits, GPU node selector/toleration, and manifest rendering are implemented and
  tested, and the GPU manifest passed a server-side dry run, but no GPU throughput is claimed.
- MPS has no vendor utilization counter. The per-run `utilization_pct` for MPS is the training
  process's CPU utilization (clearly labeled in the API/dashboard); memory comes from `torch.mps`.
- The kind development manifests use `emptyDir` for PostgreSQL/Redis/RabbitMQ and `hostPath` for
  data/artifacts. Production should use managed databases or PVCs with backups and an RWX/object-store
  artifact backend.
- The content-addressed store is a local filesystem. Next step: S3/GCS backend with the same SHA-256
  keys and lineage rows.
- One worker is enough for the demo; horizontal scaling is supported by the claim design but has not
  been stress-tested beyond the 1/2/4-worker benchmark.
- Leaderboard invalidation is explicit on metric writes plus a 30s TTL. Cross-process cache stampedes
  are possible under extreme write rates; a single-flight lock is a future improvement.
- The UI assumes one operator and a build-time API key. Add SSO, per-user keys, and role-based
  visibility before multi-tenant use.
- Training entry points and the Kubernetes executor are exercised by integration/e2e tests but are
  excluded from the coverage percentage to avoid counting subprocess-only code; coverage is 85.81% on
  the measured control-plane modules.

## 9. Dashboard screenshots

Real dashboard against the running API (local executor sweep and kind cluster):

![Runs dashboard with real sweep runs and leaderboard](docs/screenshots/runs-dashboard.png)

![Run detail with real metric curves, device samples, artifacts, and live logs](docs/screenshots/run-detail.png)

![Live log viewer streaming persisted run output](docs/screenshots/live-logs.png)

![Experiment comparison with overlaid curves and sortable diff table](docs/screenshots/compare.png)

![Submit-run form with validated hyperparameters, snapshot picker, GPU count, and idempotency key](docs/screenshots/submit-run.png)

![Queue health with work, retry, and DLQ depths](docs/screenshots/queue-health.png)

![Runs executed by the Kubernetes executor on kind](docs/screenshots/kubernetes-runs.png)

![Kubernetes run detail with its real training metrics](docs/screenshots/kubernetes-run-detail.png)

## License

MIT. See [LICENSE](LICENSE). Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).
