#!/bin/bash
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENGINE_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
KVCACHED_DIR=$(cd "$ENGINE_DIR/.." && pwd)

MODEL=meta-llama/Llama-3.1-8B
VLLM_PORT=12346
SGL_PORT=30000

op=$1

if [ "$op" == "vllm" ]; then
    source "$ENGINE_DIR/vllm-v0.9.2/.venv/bin/activate"
    export VLLM_USE_V1=1
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=VLLM
    vllm serve "$MODEL" \
    --disable-log-requests \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.5 \
    --port="$VLLM_PORT"
elif [ "$op" == "sgl" -o "$op" == "sglang" ]; then
    source "$ENGINE_DIR/sglang-v0.4.6.post2/.venv/bin/activate"
    export ENABLE_KVCACHED=true
    export KVCACHED_IPC_NAME=SGLANG
    python -m sglang.launch_server --model "$MODEL" \
    --disable-radix-cache \
    --trust-remote-code \
    --mem-fraction-static 0.5 \
    --port "$SGL_PORT"
else
    echo "Invalid option: $op"
    exit 1
fi
