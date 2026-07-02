#!/usr/bin/env bash
set -uo pipefail

ENV="crafter"
START_CODEBASE="scaffolds/inner_agent_v0"
NUM_ITERATIONS=5
MAX_FRESH_ATTEMPTS=3
MAX_CONTINUATION_RETRIES=1

META_MODEL="opus"
META_EFFORT="max"
META_MAX_TURNS=300
META_TIMEOUT=3600

MODEL_ID="Qwen/Qwen2.5-32B-Instruct"
GPU=0
VLLM_PORT=2026
VLLM_GPU_UTIL=0.98
VLLM_BASE_URL="http://0.0.0.0:${VLLM_PORT}/v1"

EVAL_SEEDS="42,43,44,45,46,47,48,49,50,51"
NUM_EVAL_EPISODES=10
NUM_EVAL_WORKERS=10

EVAL_CONDA_ENV="balrog"

WORK_DIR="loop1_output_${ENV}"

eval "$(conda shell.bash hook)"
conda activate "$EVAL_CONDA_ENV"

read_metric() {
    python3 -c "import json,sys; print(json.load(open(sys.argv[1]+'/summary.json'))['average_progress'])" "$1" 2>/dev/null || echo ""
}
improved() { awk "BEGIN{ exit !($1 > $2) }"; }
copy_clean() {
    rsync -a --exclude='__pycache__' --exclude='logs/' --exclude='results/' --exclude='memory/' "$1/" "$2/"
}

run_eval() {
    local cb="$1" run="$2"
    ( cd "$cb" && python eval.py \
        agent.type=memory \
        eval.run_name="$run" \
        "eval.seeds=[${EVAL_SEEDS}]" \
        "eval.num_episodes.${ENV}=${NUM_EVAL_EPISODES}" \
        eval.num_workers="$NUM_EVAL_WORKERS" \
        client.client_name=vllm \
        client.model_id="$MODEL_ID" \
        client.base_url="$VLLM_BASE_URL" \
        envs.names="$ENV" \
        agent.memory_dir="memory/${run}" )
    RESULTS_DIR="${cb}/results/${run}"
    LOG_FILE="${RESULTS_DIR}/eval.log"
    METRIC="$(read_metric "$RESULTS_DIR")"
}

propose_apply_eval() {
    local out="$1" new_cb="$2" run="$3" cont_results="${4:-}" cont_log="${5:-}"
    mkdir -p "$out"
    local cmd=( python loop1_scaffold_evolution/meta_loop.py
        --env "$ENV" --codebase-dir "$CUR_CB" --results-dir "$CUR_RESULTS" --log-file "$CUR_LOG"
        --output-dir "$out" --model "$META_MODEL" --effort "$META_EFFORT"
        --max-turns "$META_MAX_TURNS" --timeout "$META_TIMEOUT" )
    if [[ -n "$cont_results" ]]; then
        cmd+=( --continue-results-dir "$cont_results" --continue-log-file "$cont_log" )
    else
        [[ -n "$PREV_LOG" && -f "$PREV_LOG" ]] && cmd+=( --prev-log-file "$PREV_LOG" )
        [[ -n "$CUR_MD" && -f "$CUR_MD" ]] && cmd+=( --method-description "$CUR_MD" )
    fi
    "${cmd[@]}"

    local revised="${out}/output/revised_code"
    [[ -d "$revised" && -n "$(ls -A "$revised" 2>/dev/null)" ]] || return 1

    echo "=== (2) apply the revision to a fresh codebase copy ==="
    rm -rf "$new_cb"; copy_clean "$CUR_CB" "$new_cb"
    ( cd "$revised" && find . -type f -not -path '*/__pycache__/*' ) | while read -r f; do
        rel="${f#./}"; mkdir -p "$(dirname "${new_cb}/${rel}")"; cp "${revised}/${rel}" "${new_cb}/${rel}"
    done

    echo "=== (3) evaluate the revised scaffold (same base model, fixed seeds) ==="
    run_eval "$new_cb" "$run"
}

mkdir -p "$WORK_DIR"

echo "=== serve base model: ${MODEL_ID} on GPU ${GPU}, port ${VLLM_PORT} ==="
CUDA_VISIBLE_DEVICES=$GPU nohup vllm serve "$MODEL_ID" \
    --port "$VLLM_PORT" --served-model-name "$MODEL_ID" \
    --gpu-memory-utilization "$VLLM_GPU_UTIL" > vllm_base.log 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null' EXIT
until grep -q "Application startup complete" vllm_base.log 2>/dev/null; do sleep 15; done
echo "  base model serving at ${VLLM_BASE_URL}"

echo "=== (0) evaluate the starting scaffold: ${START_CODEBASE} ==="
CUR_CB="$START_CODEBASE"
run_eval "$CUR_CB" "${ENV}_base"
CUR_RESULTS="$RESULTS_DIR"; CUR_LOG="$LOG_FILE"; CUR_METRIC="${METRIC:-0}"
PREV_LOG=""; CUR_MD=""
echo "  starting progression: ${CUR_METRIC}"

gate_revision() {
    echo "=== gate: keep only if progression improves (got ${METRIC:-<none>}, current best ${CUR_METRIC}) ==="
    if [[ -n "${METRIC:-}" ]] && improved "$METRIC" "$CUR_METRIC"; then
        echo "  accepted: ${CUR_METRIC} -> ${METRIC}"
        PREV_LOG="$CUR_LOG"
        CUR_CB="$NEW_CB"; CUR_RESULTS="$RESULTS_DIR"; CUR_LOG="$LOG_FILE"; CUR_METRIC="$METRIC"
        CUR_MD="${OUT}/output/method_description.md"
        ACCEPTED=true
    else
        echo "  did not improve over ${CUR_METRIC}"
        FAIL_RESULTS="$RESULTS_DIR"; FAIL_LOG="$LOG_FILE"
    fi
}

for (( i=1; i<=NUM_ITERATIONS; i++ )); do
    echo "=== revision v${i}/${NUM_ITERATIONS} ==="
    ACCEPTED=false

    for (( attempt=1; attempt<=MAX_FRESH_ATTEMPTS; attempt++ )); do
        [[ "$ACCEPTED" == "true" ]] && break
        OUT="${WORK_DIR}/v${i}_revision/attempt${attempt}"
        NEW_CB="${WORK_DIR}/agent_v${i}_attempt${attempt}"
        RUN="${ENV}_v${i}_attempt${attempt}"
        echo "=== fresh attempt ${attempt}/${MAX_FRESH_ATTEMPTS}: meta-LLM reviews traces and writes a revised scaffold ==="
        if ! propose_apply_eval "$OUT" "$NEW_CB" "$RUN" "" ""; then
            echo "  no revised code produced; ending this revision"
            break
        fi
        gate_revision
        [[ "$ACCEPTED" == "true" ]] && break

        for (( continuation=1; continuation<=MAX_CONTINUATION_RETRIES; continuation++ )); do
            OUT="${WORK_DIR}/v${i}_revision/attempt${attempt}_continue${continuation}"
            NEW_CB="${WORK_DIR}/agent_v${i}_attempt${attempt}_continue${continuation}"
            RUN="${ENV}_v${i}_attempt${attempt}_continue${continuation}"
            echo "=== continuation ${continuation}/${MAX_CONTINUATION_RETRIES}: meta-LLM revises again, fed the failed revison ==="
            if ! propose_apply_eval "$OUT" "$NEW_CB" "$RUN" "$FAIL_RESULTS" "$FAIL_LOG"; then
                echo "  no revised code produced; ending this revision"
                break
            fi
            gate_revision
            [[ "$ACCEPTED" == "true" ]] && break
        done
    done

    [[ "$ACCEPTED" == "true" ]] || \
        echo "=== revision v${i}: no improving revision after all attempts; keeping the current scaffold (${CUR_METRIC}) ==="
done

echo "=== done. best scaffold: ${CUR_CB} (progression ${CUR_METRIC}) ==="
