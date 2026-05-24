#!/usr/bin/env bash
# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# 只跑 flexprefill：

#   ATTN_TYPES=flexprefill bash MInference/experiments/needle_in_a_haystack/
#   run_multi_attn.sh

set -euo pipefail

export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINFERENCE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${MINFERENCE_ROOT}"

# Usage:
#   bash experiments/needle_in_a_haystack/run_multi_attn.sh
#   ATTN_TYPES=minference,inf_llm,dense,flexprefill MAX_LENGTH=100000 bash experiments/needle_in_a_haystack/run_multi_attn.sh
#   DRY_RUN=1 bash experiments/needle_in_a_haystack/run_multi_attn.sh
#
# Extra arguments are forwarded to needle_test.py, for example:
#   bash experiments/needle_in_a_haystack/run_multi_attn.sh --trust_remote_code

MODEL_NAME="${MODEL_NAME:-gradientai/Llama-3-8B-Instruct-Gradient-1048k}"
MODEL_TAG="${MODEL_TAG:-LLaMA_1M}"
ATTN_TYPES="${ATTN_TYPES:-minference,inf_llm,dense,flexprefill}"
MAX_LENGTH="${MAX_LENGTH:-1000000}"
MIN_LENGTH="${MIN_LENGTH:-1000}"
ROUNDS="${ROUNDS:-5}"
OUTPUT_PATH="${OUTPUT_PATH:-./needle}"
JOBS="${JOBS:-0-4}"
PATTERN_PATH="${PATTERN_PATH:-}"
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"
KV_CACHE_CPU="${KV_CACHE_CPU:-0}"
KV_CACHE_CPU_DEVICE="${KV_CACHE_CPU_DEVICE:-cpu}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" ]]; then
    mkdir -p data
fi
if [[ "${DRY_RUN}" != "1" && "${DRY_RUN}" != "true" && ! -f data/pg19_mini.jsonl ]]; then
    wget https://github.com/liyucheng09/LatestEval/releases/download/pg19/pg19_mini.jsonl -O ./data/pg19_mini.jsonl
fi

IFS=',' read -r -a methods <<< "${ATTN_TYPES}"

for method in "${methods[@]}"; do
    method="${method//[[:space:]]/}"
    case "${method}" in
        minference)
            attn_type="minference"
            kv_type="dense"
            mode_name="minference"
            ;;
        inf_llm)
            attn_type="inf_llm"
            kv_type="dense"
            mode_name="inf_llm"
            ;;
        a_shape)
            attn_type="a_shape"
            kv_type="dense"
            mode_name="a_shape"
            ;;
        dense)
            attn_type="dense"
            kv_type="dense"
            mode_name="dense"
            ;;
        flexprefill)
            attn_type="flexprefill"
            kv_type="dense"
            mode_name="flexprefill"
            ;;
        *)
            echo "Unsupported method '${method}'. Use one of: minference, inf_llm, streamingllm, dense, flexprefill, a_shape."
            exit 1
            ;;
    esac

    run_name="${RUN_NAME_PREFIX}${mode_name}_${MODEL_TAG}"
    echo "Running ${mode_name}: --attn_type ${attn_type} --kv_type ${kv_type} --run_name ${run_name}"

    cmd=(
        python experiments/needle_in_a_haystack/needle_test.py
        --model_name "${MODEL_NAME}"
        --max_length "${MAX_LENGTH}"
        --min_length "${MIN_LENGTH}"
        --rounds "${ROUNDS}"
        --attn_type "${attn_type}"
        --kv_type "${kv_type}"
        --output_path "${OUTPUT_PATH}"
        --run_name "${run_name}"
        --kv_cache_cpu_device "${KV_CACHE_CPU_DEVICE}"
    )

    if [[ -n "${JOBS}" ]]; then
        cmd+=(--jobs "${JOBS}")
    fi
    if [[ -n "${PATTERN_PATH}" ]]; then
        cmd+=(--pattern_path "${PATTERN_PATH}")
    fi
    if [[ "${KV_CACHE_CPU}" == "1" || "${KV_CACHE_CPU}" == "true" ]]; then
        cmd+=(--kv_cache_cpu)
    fi
    if [[ "$#" -gt 0 ]]; then
        cmd+=("$@")
    fi

    if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" ]]; then
        printf 'DRY_RUN:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
done
