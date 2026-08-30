#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

echo "=== 1. Pulling latest changes from origin/main ==="
git fetch origin main
git reset --hard origin/main

echo "=== 2. Restarting SignTrustMap AI Service ==="
sudo systemctl restart signtrustmap-ai

echo "=== 3. Verifying Health Check ==="
sleep 5

MAX_RETRIES=30
HEALTH_URL="http://127.0.0.1:8000/health"

for i in $(seq 1 $MAX_RETRIES); do
    if curl -s -f "$HEALTH_URL" > /dev/null; then
        echo "✅ Deployment successful! Service is healthy on $HEALTH_URL"
        exit 0
    fi
    echo "Waiting for service to become healthy... ($i/$MAX_RETRIES)"
    sleep 3
done

echo "❌ Deployment verification failed: Service did not become healthy in time."
exit 1
