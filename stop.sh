#!/usr/bin/env sh
# Stops and uninstalls the local AI agent: removes containers, named
# volumes (Ollama models + WhisperX cache + RAG indexes inside the volume),
# networks, and the locally-built images for this project. Pulled images
# (e.g. ollama/ollama:latest) are kept. Asks for confirmation because it's
# destructive — restart with start.sh to set up again from scratch.

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

printf 'Stop and uninstall everything (containers, Ollama models, WhisperX cache, RAG indexes, built images)? [y/N] '
read -r CONFIRM
case "$CONFIRM" in
    y|Y|yes|YES) ;;
    *) echo "Aborted."; exit 0 ;;
esac

# --profile voice ensures the whisperx service is included if it was created.
docker compose --profile voice down -v --rmi local --remove-orphans

# Clear voice preference so the next start.sh run re-asks the user.
if [ -f "$ENV_FILE" ]; then
    grep -v '^ENABLE_VOICE=' "$ENV_FILE" > "$ENV_FILE.tmp" || true
    mv "$ENV_FILE.tmp" "$ENV_FILE"
fi

echo
echo "Uninstalled. Run start.sh to set up again from scratch."
