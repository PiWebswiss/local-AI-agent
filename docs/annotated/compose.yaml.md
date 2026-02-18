# compose.yaml — line-by-line explanation

Generated: 2026-02-18T09:56:40

Purpose: Docker Compose stack (agent + Ollama) with hardening and no published ports.

Format: each original source line is shown with its line number, followed by a short explanation.

```text
  1 services:
      ↳ Starts a new block (indented section) in Python.
  2   ollama:
      ↳ Starts a new block (indented section) in Python.
  3     image: ollama/ollama:latest
      ↳ Implementation detail: part of the surrounding logic.
  4     gpus: all
      ↳ Implementation detail: part of the surrounding logic.
  5     volumes:
      ↳ Starts a new block (indented section) in Python.
  6       - ollama:/root/.ollama
      ↳ Implementation detail: part of the surrounding logic.
  7     restart: unless-stopped
      ↳ Implementation detail: part of the surrounding logic.
  8     networks:
      ↳ Starts a new block (indented section) in Python.
  9       - backend
      ↳ Implementation detail: part of the surrounding logic.
 10       - egress
      ↳ Implementation detail: part of the surrounding logic.
 11     healthcheck:
      ↳ Starts a new block (indented section) in Python.
 12       test: ["CMD-SHELL", "ollama list >/dev/null 2>&1 || exit 1"]
      ↳ Implementation detail: part of the surrounding logic.
 13       interval: 5s
      ↳ Implementation detail: part of the surrounding logic.
 14       timeout: 5s
      ↳ Implementation detail: part of the surrounding logic.
 15       retries: 30
      ↳ Implementation detail: part of the surrounding logic.
 16 
      ↳ Blank line for readability.
 17   ollama_init:
      ↳ Starts a new block (indented section) in Python.
 18     image: ollama/ollama:latest
      ↳ Implementation detail: part of the surrounding logic.
 19     depends_on:
      ↳ Starts a new block (indented section) in Python.
 20       ollama:
      ↳ Starts a new block (indented section) in Python.
 21         condition: service_healthy
      ↳ Implementation detail: part of the surrounding logic.
 22     networks:
      ↳ Starts a new block (indented section) in Python.
 23       - backend
      ↳ Implementation detail: part of the surrounding logic.
 24       - egress
      ↳ Implementation detail: part of the surrounding logic.
 25     environment:
      ↳ Starts a new block (indented section) in Python.
 26       OLLAMA_HOST: http://ollama:11434
      ↳ Implementation detail: part of the surrounding logic.
 27       OLLAMA_MODEL: ${OLLAMA_MODEL:-gemma3:1b}
      ↳ Implementation detail: part of the surrounding logic.
 28     entrypoint: ["/bin/sh", "-lc", "ollama pull \"$$OLLAMA_MODEL\""]
      ↳ Implementation detail: part of the surrounding logic.
 29     restart: "no"
      ↳ Implementation detail: part of the surrounding logic.
 30 
      ↳ Blank line for readability.
 31   agent:
      ↳ Starts a new block (indented section) in Python.
 32     build: .
      ↳ Implementation detail: part of the surrounding logic.
 33     depends_on:
      ↳ Starts a new block (indented section) in Python.
 34       ollama_init:
      ↳ Starts a new block (indented section) in Python.
 35         condition: service_completed_successfully
      ↳ Implementation detail: part of the surrounding logic.
 36     networks:
      ↳ Starts a new block (indented section) in Python.
 37       - backend
      ↳ Implementation detail: part of the surrounding logic.
 38       - egress
      ↳ Implementation detail: part of the surrounding logic.
 39     environment:
      ↳ Starts a new block (indented section) in Python.
 40       HOME: /tmp
      ↳ Implementation detail: part of the surrounding logic.
 41       XDG_CACHE_HOME: /tmp/.cache
      ↳ Implementation detail: part of the surrounding logic.
 42       XDG_CONFIG_HOME: /tmp/.config
      ↳ Implementation detail: part of the surrounding logic.
 43       XDG_DATA_HOME: /tmp/.local/share
      ↳ Implementation detail: part of the surrounding logic.
 44       AGENT_RAG_DIR: /rag
      ↳ Implementation detail: part of the surrounding logic.
 45       AGENT_OUT_DIR: /files
      ↳ Implementation detail: part of the surrounding logic.
 46       OLLAMA_HOST: http://ollama:11434
      ↳ Implementation detail: part of the surrounding logic.
 47       OLLAMA_MODEL: ${OLLAMA_MODEL:-gemma3:1b}
      ↳ Implementation detail: part of the surrounding logic.
 48       OLLAMA_MAX_B: ${OLLAMA_MAX_B:-4}
      ↳ Implementation detail: part of the surrounding logic.
 49       OLLAMA_TIMEOUT_S: ${OLLAMA_TIMEOUT_S:-120}
      ↳ Implementation detail: part of the surrounding logic.
 50       OCR_SPACE_API_KEY: ${OCR_SPACE_API_KEY:-}
      ↳ Implementation detail: part of the surrounding logic.
 51       OCR_SPACE_LANGUAGE: ${OCR_SPACE_LANGUAGE:-eng}
      ↳ Implementation detail: part of the surrounding logic.
 52       OCR_SPACE_TIMEOUT_S: ${OCR_SPACE_TIMEOUT_S:-60}
      ↳ Implementation detail: part of the surrounding logic.
 53       OCR_SPACE_MAX_BYTES: ${OCR_SPACE_MAX_BYTES:-8000000}
      ↳ Implementation detail: part of the surrounding logic.
 54     read_only: true
      ↳ Implementation detail: part of the surrounding logic.
 55     tmpfs:
      ↳ Starts a new block (indented section) in Python.
 56       - /tmp
      ↳ Implementation detail: part of the surrounding logic.
 57     cap_drop:
      ↳ Starts a new block (indented section) in Python.
 58       - ALL
      ↳ Implementation detail: part of the surrounding logic.
 59     security_opt:
      ↳ Starts a new block (indented section) in Python.
 60       - no-new-privileges:true
      ↳ Implementation detail: part of the surrounding logic.
 61     init: true
      ↳ Implementation detail: part of the surrounding logic.
 62     stdin_open: true
      ↳ Implementation detail: part of the surrounding logic.
 63     tty: true
      ↳ Implementation detail: part of the surrounding logic.
 64     volumes:
      ↳ Starts a new block (indented section) in Python.
 65       - ./files:/files
      ↳ Implementation detail: part of the surrounding logic.
 66       - ./rag:/rag
      ↳ Implementation detail: part of the surrounding logic.
 67     command: ["python", "agent.py", "chat"]
      ↳ Implementation detail: part of the surrounding logic.
 68     restart: "no"
      ↳ Implementation detail: part of the surrounding logic.
 69 
      ↳ Blank line for readability.
 70 volumes:
      ↳ Starts a new block (indented section) in Python.
 71   ollama:
      ↳ Starts a new block (indented section) in Python.
 72 
      ↳ Blank line for readability.
 73 networks:
      ↳ Starts a new block (indented section) in Python.
 74   backend:
      ↳ Starts a new block (indented section) in Python.
 75     internal: true
      ↳ Implementation detail: part of the surrounding logic.
 76   egress:
      ↳ Starts a new block (indented section) in Python.
```
