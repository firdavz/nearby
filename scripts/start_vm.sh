#!/usr/bin/env bash
# Start Second Sight on the lab VM with the environment W&B and Cosmos need.  Owner: D.
#   bash scripts/start_vm.sh            (from the repo root; tunnel localhost:8000 to your laptop)
# Anything already exported in your shell beats .env (load_dotenv never overrides), so the W&B project
# is forced into the entity/project form here, which is what W&B Inference's OpenAI-Project header wants.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] && source .venv/bin/activate
export WANDB_TEAM="${WANDB_TEAM:-vastdata}"
case "${WANDB_PROJECT:-team-48}" in
  */*) export WANDB_PROJECT ;;
  *)   export WANDB_PROJECT="${WANDB_TEAM}/${WANDB_PROJECT:-team-48}" ;;
esac
echo "WANDB_PROJECT=$WANDB_PROJECT  COSMOS_FPS=${COSMOS_FPS:-from .env}"
exec uvicorn main:app --host 127.0.0.1 --port "${PORT:-8000}" "$@"
