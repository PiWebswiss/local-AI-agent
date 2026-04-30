#!/usr/bin/env sh
# One-command launcher for the local AI agent. Asks the user once whether
# to enable voice input (WhisperX), persists the choice to .env, and then
# runs `docker compose up -d --build` with the right profile so WhisperX
# is only built/installed when the user actually wants it.

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

VOICE_PREF=""
if [ -f "$ENV_FILE" ]; then
    VOICE_PREF=$(grep -E '^ENABLE_VOICE=' "$ENV_FILE" | head -1 | cut -d= -f2- || true)
fi

if [ -z "$VOICE_PREF" ]; then
    printf '\n'
    printf 'This project includes optional voice input via WhisperX.\n'
    printf 'Enabling it adds ~1.5 GB to the Docker build and downloads model weights on first start.\n'
    printf 'You can change this later by re-running this script (after deleting the\n'
    printf 'ENABLE_VOICE line from .env) or from the web UI Settings tab.\n\n'
    printf 'Enable voice input? [y/N] '
    read -r ANSWER

    case "$ANSWER" in
        y|Y|yes|YES) VOICE_PREF="true" ;;
        *)           VOICE_PREF="false" ;;
    esac

    # Persist preference to .env so future runs don't ask again.
    [ -f "$ENV_FILE" ] || : > "$ENV_FILE"
    printf '\nENABLE_VOICE=%s\n' "$VOICE_PREF" >> "$ENV_FILE"
    printf 'Saved preference: ENABLE_VOICE=%s\n' "$VOICE_PREF"
fi

if [ "$VOICE_PREF" = "true" ]; then
    export COMPOSE_PROFILES=voice
    echo "Starting with voice input enabled..."
else
    unset COMPOSE_PROFILES
    echo "Starting without voice input..."
fi

docker compose up -d --build
