#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

MODEL="${VLLM_MODEL:-defog/sqlcoder-7b-2}"
CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-}"
if [[ -z "$CHAT_TEMPLATE" && "$MODEL" == defog/sqlcoder* ]]; then
    CHAT_TEMPLATE="scripts/sqlcoder_chat_template.jinja"
fi

CMD=(uv run python -m vllm.entrypoints.openai.api_server
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90)

if [[ -n "$CHAT_TEMPLATE" ]]; then
    CMD+=(--chat-template "$CHAT_TEMPLATE")
fi

exec "${CMD[@]}"
