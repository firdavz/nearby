#!/usr/bin/env bash
# Person C: switch on the real brain (W&B LLM) and the natural voice (ElevenLabs) on THIS machine.
# Keys are typed here, hidden, and written only to .env (gitignored). Never put them in chat or git.
#
#   bash scripts/enable_brain.sh
#
# Then restart uvicorn. Finishes by running scripts/doctor.py.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
[ -f .env ] || cp .env.example .env

setkv() {   # setkv KEY VALUE: replace the KEY= line in .env, or append it. Done in Python so keys need no escaping.
  "$PY" - "$1" "$2" <<'EOF'
import pathlib, sys
k, v = sys.argv[1], sys.argv[2]
p = pathlib.Path(".env")
lines = p.read_text().splitlines()
if any(l.startswith(k + "=") for l in lines):
    lines = [f"{k}={v}" if l.startswith(k + "=") else l for l in lines]
else:
    lines.append(f"{k}={v}")
p.write_text("\n".join(lines) + "\n")
EOF
}

echo "W&B API key: https://wandb.ai/authorize  (input hidden; Enter keeps the current one)"
read -rsp "  WANDB_API_KEY: " WK; echo
[ -n "$WK" ] && setkv WANDB_API_KEY "$WK"

echo "W&B project as entity/project for the Weave traces (Enter = the key's default)"
read -rp "  WANDB_PROJECT: " WP
setkv WANDB_PROJECT "$WP"

echo "ElevenLabs key for the natural voice (input hidden; Enter skips = browser voice)"
read -rsp "  ELEVENLABS_API_KEY: " EK; echo
[ -n "$EK" ] && setkv ELEVENLABS_API_KEY "$EK"

setkv USE_REAL_LLM 1
setkv AGENT_MODEL nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B
setkv SPEAK_COOLDOWN_S 10
"$PY" -m pip install --quiet openai weave httpx

echo
"$PY" scripts/doctor.py || true
echo
echo "Restart the server so it picks this up:  uvicorn main:app --reload --port 8000"
