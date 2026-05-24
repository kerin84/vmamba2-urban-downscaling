#!/usr/bin/env bash
# deploy_to_server.sh — sync repo code to server and run build-static
#
# Usage:
#   ./scripts/deploy_to_server.sh <user@host> <server_data_dir>
#
# Example:
#   ./scripts/deploy_to_server.sh kerin@192.168.1.100 /data/weather_v2
#
# What it does:
#   1. Rsyncs code (no data, no experiments, no .git) to the server
#   2. Sets DATA_DIR on the server
#   3. Runs docker compose build-static
#   4. Tails the logs

set -euo pipefail

SERVER="${1:?Usage: $0 <user@host> <server_data_dir>}"
SERVER_DATA_DIR="${2:?Usage: $0 <user@host> <server_data_dir>}"
SERVER_REPO_DIR="${3:-/tmp/weather_urban_downscaling_v2}"

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Deploying to $SERVER ==="
echo "  Repo → $SERVER_REPO_DIR"
echo "  Data → $SERVER_DATA_DIR"
echo ""

# 1. Sync code (exclude data, experiments, __pycache__, .venv)
echo "Syncing code..."
rsync -avz --delete \
    --exclude='.git' \
    --exclude='data/' \
    --exclude='experiments/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.venv/' \
    "$SCRIPT_DIR/" \
    "$SERVER:$SERVER_REPO_DIR/"

echo ""
echo "Running build-static on server..."
ssh "$SERVER" bash -lc "
    set -e
    cd $SERVER_REPO_DIR
    export DATA_DIR=$SERVER_DATA_DIR
    docker compose -f docker/compose.yml --profile static up --build
"
