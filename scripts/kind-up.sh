#!/usr/bin/env bash
# Create a kind cluster, build/load the TrainTrack image, migrate Postgres,
# register the dataset snapshot, and deploy the API and worker.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLUSTER="${KIND_CLUSTER:-traintrack}"
NAMESPACE="traintrack"
IMAGE="traintrack:dev"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

echo "==> TrainTrack root: $ROOT"
echo "==> Preparing shared dataset/artifact directories (dev permissions)"
mkdir -p "$ROOT/data" "$ROOT/artifacts"
chmod -R a+rwX "$ROOT/data" "$ROOT/artifacts"

if ! kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  echo "==> Creating kind cluster $CLUSTER"
  sed "s|__PROJECT_ROOT__|$ROOT|g" "$ROOT/deploy/k8s/kind-config.yaml" > /tmp/traintrack-kind-config.yaml
  kind create cluster --name "$CLUSTER" --config /tmp/traintrack-kind-config.yaml --wait 180s
else
  echo "==> Reusing kind cluster $CLUSTER"
fi

echo "==> Building $IMAGE"
docker build -t "$IMAGE" "$ROOT"

echo "==> Loading images into kind"
kind load docker-image "$IMAGE" --name "$CLUSTER"
# Base images are pulled by the kubelet. Preloading multi-arch images through
# containerd can fail on Docker's containerd image store ("content digest not
# found"), so only the locally built image is loaded explicitly.

echo "==> Applying namespace, config, data services"
kubectl apply -f "$ROOT/deploy/k8s/namespace.yaml"
kubectl apply -f "$ROOT/deploy/k8s/secret.yaml"
kubectl apply -f "$ROOT/deploy/k8s/configmap.yaml"
kubectl apply -f "$ROOT/deploy/k8s/postgres.yaml"
kubectl apply -f "$ROOT/deploy/k8s/redis.yaml"
kubectl apply -f "$ROOT/deploy/k8s/rabbitmq.yaml"
kubectl apply -f "$ROOT/deploy/k8s/rbac.yaml"

kubectl -n "$NAMESPACE" rollout status deployment/postgres --timeout=180s
kubectl -n "$NAMESPACE" rollout status deployment/redis --timeout=180s
kubectl -n "$NAMESPACE" rollout status deployment/rabbitmq --timeout=240s

echo "==> Running Alembic migrations inside the cluster"
kubectl -n "$NAMESPACE" delete job traintrack-migrate --ignore-not-found >/dev/null
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: traintrack-migrate
  namespace: $NAMESPACE
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: migrate
          image: $IMAGE
          imagePullPolicy: IfNotPresent
          command: ["alembic", "upgrade", "head"]
          envFrom:
            - configMapRef:
                name: traintrack-config
EOF
kubectl -n "$NAMESPACE" wait --for=condition=complete job/traintrack-migrate --timeout=300s

echo "==> Registering the MNIST dataset snapshot in the cluster registry"
kubectl -n "$NAMESPACE" delete job traintrack-snapshot --ignore-not-found >/dev/null
kubectl apply -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: traintrack-snapshot
  namespace: $NAMESPACE
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: snapshot
          image: $IMAGE
          imagePullPolicy: IfNotPresent
          command: ["python", "-m", "traintrack.pipeline.preprocess", "--version", "mnist-v1"]
          envFrom:
            - configMapRef:
                name: traintrack-config
          volumeMounts:
            - name: data
              mountPath: /data
      volumes:
        - name: data
          hostPath:
            path: /data
            type: DirectoryOrCreate
EOF
kubectl -n "$NAMESPACE" wait --for=condition=complete job/traintrack-snapshot --timeout=600s

echo "==> Deploying API and worker"
kubectl apply -f "$ROOT/deploy/k8s/api.yaml"
kubectl apply -f "$ROOT/deploy/k8s/worker.yaml"
kubectl -n "$NAMESPACE" rollout status deployment/traintrack-api --timeout=300s
kubectl -n "$NAMESPACE" rollout status deployment/traintrack-worker --timeout=300s

echo "==> Validating the GPU request manifest with a server-side dry run"
"$PYTHON" "$ROOT/scripts/render_gpu_manifest.py" --output /tmp/traintrack-gpu-job.yaml
kubectl apply --dry-run=server -f /tmp/traintrack-gpu-job.yaml

echo
echo "TrainTrack is running on kind."
echo "API:      kubectl -n $NAMESPACE port-forward svc/traintrack-api 18011:8010"
echo "Worker:   kubectl -n $NAMESPACE logs deploy/traintrack-worker -f"
echo "Runs:     python scripts/run_k8s_experiments.py --api-url http://localhost:18011"
