#!/usr/bin/env bash
set -uo pipefail

ENV="crafter"
ENV_DIR="crafter"
RUN_NAME="train_engine"

CODEBASE_DATA="scaffolds/crafter_v5"
CODEBASE_EVAL="scaffolds/crafter_v5_twomodel"

NUM_ITERATIONS=5
MAX_CONFIGS_PER_ITER=2
BASELINE_METRIC=""

NUM_TRAINING_EPISODES=100
NUM_DATA_WORKERS=10

CC_MODEL="opus"
CC_EFFORT="max"
CC_MAX_TURNS=300
CC_TIMEOUT=3600

MODEL_ID="Qwen/Qwen2.5-32B-Instruct"
LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-}"

GPU0=0
GPU1=1
PORT_MEMORY=20260
PORT_GAMEPLAY=20261
VLLM_GPU_UTIL=0.98

EVAL_SEEDS="42,43,44,45,46,47,48,49,50,51"
NUM_EVAL_EPISODES=10
NUM_EVAL_WORKERS=10

TRAIN_CONDA_ENV="llamafactory"
EVAL_CONDA_ENV="balrog"

ENGINE_DIR="loop2_output_${ENV}/${RUN_NAME}"
DATA_ENGINE_PRISTINE="loop2_training_engine/data_engine.py"
DATA_ENGINE_WORK="${ENGINE_DIR}/data_engine.py"
POSTPROCESS="loop2_training_engine/postprocess.py"
CONFIG_RECS="loop2_training_engine/env/${ENV_DIR}/config_recommendation.md"
DECIDER="loop2_training_engine/training_engine.py"
LEDGER_PY="loop2_training_engine/ledger.py"
LEDGER="${ENGINE_DIR}/ledger.jsonl"

TRAIN_DATA_RUN_NAME="${RUN_NAME}_traindata"
RESULTS_DIR="${CODEBASE_DATA}/results/${TRAIN_DATA_RUN_NAME}"
MEMORY_DIR="${CODEBASE_DATA}/memory/${TRAIN_DATA_RUN_NAME}"

eval "$(conda shell.bash hook)"

[[ -z "$LLAMA_FACTORY_DIR" ]] && { echo "set LLAMA_FACTORY_DIR"; exit 1; }
[[ ! -d "$CODEBASE_DATA" ]] && { echo "missing data codebase: $CODEBASE_DATA"; exit 1; }
[[ ! -d "$CODEBASE_EVAL" ]] && { echo "missing eval codebase: $CODEBASE_EVAL"; exit 1; }

alias_to_target() {
    case "$1" in
        ""|ALL) echo "all" ;;
        ATTN)   echo "q_proj,k_proj,v_proj,o_proj" ;;
        QV)     echo "q_proj,v_proj" ;;
        MLP)    echo "gate_proj,up_proj,down_proj" ;;
        *)      echo "$1" ;;
    esac
}

random_training_seeds() {
    python3 -c "
import random
random.seed(999)
fixed=set(int(s) for s in '${EVAL_SEEDS}'.split(','))
seeds=[]
while len(seeds) < ${NUM_TRAINING_EPISODES}:
    s=random.randint(1000,99999)
    if s not in fixed and s not in set(seeds): seeds.append(s)
print(','.join(str(s) for s in seeds))
"
}

serve_base() {
    CUDA_VISIBLE_DEVICES=$GPU1 nohup vllm serve "$MODEL_ID" --port "$PORT_GAMEPLAY" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" --served-model-name "$MODEL_ID" > vllm_gameplay.log 2>&1 &
    BASE_PID=$!
    until grep -q "Application startup complete" vllm_gameplay.log 2>/dev/null; do sleep 15; done
}

run_config() {
    local iter="$1" data_source="$2" cfg="$3" selected="$4" tag="$5"
    IFS=':' read -r R A D LR E B GA CL LRS WR SC NG TARGET <<< "$cfg"
    local lora_target; lora_target="$(alias_to_target "$TARGET")"
    local run="${ENV}_${RUN_NAME}_iter${iter}_${tag}"
    local dataset="${run}"
    local adapter="${LLAMA_FACTORY_DIR}/saves/${dataset}_lora"
    local merged="${LLAMA_FACTORY_DIR}/models/${dataset}_merged"
    local train_yaml="${LLAMA_FACTORY_DIR}/examples/train_lora/${dataset}.yaml"
    local merge_yaml="${LLAMA_FACTORY_DIR}/examples/merge_lora/${dataset}_merge.yaml"
    local served="${dataset}_trained"
    local arch="$(pwd)/${ENGINE_DIR}/iter${iter}/${tag}"
    mkdir -p "$arch"

    echo "  config ${tag}: ${cfg}  (rank=${R} lr=${LR} ep=${E} target=${TARGET})"

    conda activate "$TRAIN_CONDA_ENV"
    cp "$selected" "${LLAMA_FACTORY_DIR}/data/${dataset}.json"
    python3 - "$LLAMA_FACTORY_DIR" "$dataset" <<'PYEOF'
import json, os, sys
lf, name = sys.argv[1], sys.argv[2]
p = os.path.join(lf, "data", "dataset_info.json")
info = json.load(open(p))
info[name] = {"file_name": "%s.json" % name, "formatting": "sharegpt",
              "columns": {"messages": "conversations"},
              "tags": {"role_tag": "role", "content_tag": "content",
                       "user_tag": "user", "assistant_tag": "assistant"}}
json.dump(info, open(p, "w"), indent=2, ensure_ascii=False)
PYEOF

    local ds_config="${LLAMA_FACTORY_DIR}/ds_z3_automem.json"
    cat > "$ds_config" <<'DSEOF'
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "bf16": { "enabled": "auto" },
  "zero_optimization": {
    "stage": 3, "overlap_comm": true, "contiguous_gradients": true,
    "sub_group_size": 1000000000.0, "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto", "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1000000000.0, "stage3_max_reuse_distance": 1000000000.0,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
DSEOF
    mkdir -p "$(dirname "$train_yaml")" "$(dirname "$merge_yaml")"
    cat > "$train_yaml" <<EOF
model_name_or_path: ${MODEL_ID}
stage: sft
do_train: true
finetuning_type: lora
lora_rank: ${R}
lora_alpha: ${A}
lora_dropout: ${D}
lora_target: ${lora_target}
dataset: ${dataset}
template: qwen
cutoff_len: ${CL}
per_device_train_batch_size: ${B}
gradient_accumulation_steps: ${GA}
num_train_epochs: ${E}
learning_rate: ${LR}
lr_scheduler_type: ${LRS}
warmup_ratio: ${WR}
bf16: true
optim: adamw_torch
deepspeed: ${ds_config}
logging_steps: 5
save_strategy: "no"
output_dir: ${adapter}
max_grad_norm: 1.0
seed: 42
report_to: none
EOF
    ( cd "$LLAMA_FACTORY_DIR" && CUDA_VISIBLE_DEVICES=$GPU0,$GPU1 FORCE_TORCHRUN=1 llamafactory-cli train "$train_yaml" ) > "${arch}/training.log" 2>&1
    [[ ! -f "${adapter}/adapter_model.safetensors" ]] && { echo "  no adapter for ${tag}; skipping"; return 1; }

    rm -rf "$merged"
    cat > "$merge_yaml" <<EOF
model_name_or_path: ${MODEL_ID}
adapter_name_or_path: ${adapter}
template: qwen
finetuning_type: lora
export_dir: ${merged}
export_size: 5
export_device: cpu
export_legacy_format: false
EOF
    ( cd "$LLAMA_FACTORY_DIR" && CUDA_VISIBLE_DEVICES=$GPU0 llamafactory-cli export "$merge_yaml" )
    [[ ! -f "${merged}/config.json" ]] && { echo "  merge failed for ${tag}; skipping"; return 1; }
    python3 - "$merged" <<'PYEOF'
import json, os, sys
p = os.path.join(sys.argv[1], "tokenizer_config.json")
cfg = json.load(open(p))
if isinstance(cfg.get("extra_special_tokens"), list):
    del cfg["extra_special_tokens"]; json.dump(cfg, open(p, "w"), indent=2, ensure_ascii=False)
PYEOF

    conda activate "$EVAL_CONDA_ENV"
    CUDA_VISIBLE_DEVICES=$GPU0 nohup vllm serve "$merged" --port "$PORT_MEMORY" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" --served-model-name "$served" > vllm_memory.log 2>&1 &
    local pid_mem=$!
    CUDA_VISIBLE_DEVICES=$GPU1 nohup vllm serve "$MODEL_ID" --port "$PORT_GAMEPLAY" \
        --gpu-memory-utilization "$VLLM_GPU_UTIL" --served-model-name "$MODEL_ID" > vllm_gameplay.log 2>&1 &
    local pid_game=$!
    for log in vllm_memory.log vllm_gameplay.log; do
        until grep -q "Application startup complete" "$log" 2>/dev/null; do sleep 15; done
    done

    ( cd "$CODEBASE_EVAL" && python eval.py \
        agent.type=memory \
        eval.run_name="$run" \
        "eval.seeds=[${EVAL_SEEDS}]" \
        "eval.num_episodes.${ENV}=${NUM_EVAL_EPISODES}" \
        eval.num_workers="$NUM_EVAL_WORKERS" \
        client.client_name=vllm \
        client.model_id="$served" \
        client.base_url="http://0.0.0.0:${PORT_MEMORY}/v1" \
        client_gameplay.client_name=vllm \
        client_gameplay.model_id="$MODEL_ID" \
        client_gameplay.base_url="http://0.0.0.0:${PORT_GAMEPLAY}/v1" \
        envs.names="$ENV" \
        agent.memory_dir="memory/${run}" )
    kill "$pid_mem" "$pid_game" 2>/dev/null

    mkdir -p "${arch}/results" "${arch}/memory"
    cp -a "${CODEBASE_EVAL}/results/${run}" "${arch}/results/" 2>/dev/null
    cp -a "${CODEBASE_EVAL}/memory/${run}" "${arch}/memory/" 2>/dev/null
    cp "$train_yaml" "${arch}/train_lora.yaml" 2>/dev/null
    cp "$merge_yaml" "${arch}/merge_lora.yaml" 2>/dev/null
    cp "$selected" "${arch}/selected_data.json" 2>/dev/null
    local tloss="?"
    [[ -f "${adapter}/all_results.json" ]] && tloss=$(python3 -c "import json
try: print(round(float(json.load(open('${adapter}/all_results.json'))['train_loss']), 4))
except Exception: print('?')" 2>/dev/null || echo "?")
    local n_ex; n_ex=$(python3 -c "import json;print(len(json.load(open('${selected}'))))" 2>/dev/null || echo "?")
    python3 "$LEDGER_PY" --run-dir "$arch" --env "$ENV" \
        --iteration "$iter" --data-source "$data_source" --config "$cfg" \
        --num-examples "$n_ex" --train-loss "$tloss" --eval-seeds "$EVAL_SEEDS" >> "$LEDGER"
}

mkdir -p "$ENGINE_DIR"
touch "$LEDGER"
cp "$DATA_ENGINE_PRISTINE" "$DATA_ENGINE_WORK"
cp "$POSTPROCESS" "${ENGINE_DIR}/postprocess.py"

echo "=== (0) collect base-model training traces (random seeds, disjoint from eval) ==="
N_TRACES=0
[[ -d "$RESULTS_DIR" ]] && N_TRACES=$(find "$RESULTS_DIR" -name '*_run_*.json' -not -name '*_summary.json' 2>/dev/null | wc -l)
if [[ "$N_TRACES" -gt 0 ]]; then
    echo "  found ${N_TRACES} existing trace episodes; skipping collection"
else
    conda activate "$EVAL_CONDA_ENV"
    serve_base
    TRAIN_SEEDS="$(random_training_seeds)"
    ( cd "$CODEBASE_DATA" && python eval.py \
        agent.type=memory \
        eval.run_name="$TRAIN_DATA_RUN_NAME" \
        eval.save_images=False \
        "eval.seeds=[${TRAIN_SEEDS}]" \
        "eval.num_episodes.${ENV}=${NUM_TRAINING_EPISODES}" \
        eval.num_workers="$NUM_DATA_WORKERS" \
        client.client_name=vllm \
        client.model_id="$MODEL_ID" \
        client.base_url="http://0.0.0.0:${PORT_GAMEPLAY}/v1" \
        envs.names="$ENV" \
        agent.memory_dir="memory/${TRAIN_DATA_RUN_NAME}" )
    kill "$BASE_PID" 2>/dev/null
    [[ -d "$RESULTS_DIR" ]] || { echo "training-data collection produced no results"; exit 1; }
fi

for (( i=1; i<=NUM_ITERATIONS; i++ )); do
    echo "=== iteration ${i}/${NUM_ITERATIONS} ==="
    RECIPE_DIR="${ENGINE_DIR}/iter${i}/recipe"
    RECIPE="${RECIPE_DIR}/recipe.json"
    mkdir -p "$RECIPE_DIR"

    echo "=== (1) training engine decider decides recipe.json ==="
    conda activate "$EVAL_CONDA_ENV"
    python "$DECIDER" \
        --env "$ENV" --iteration "$i" \
        --output-dir "$(pwd)/$RECIPE_DIR" --ledger "$(pwd)/$LEDGER" \
        --data-engine-code "$(pwd)/$DATA_ENGINE_WORK" \
        --config-recommendations "$(pwd)/$CONFIG_RECS" \
        --baseline-metric "$BASELINE_METRIC" \
        --max-configs "$MAX_CONFIGS_PER_ITER" \
        --model "$CC_MODEL" --effort "$CC_EFFORT" \
        --max-turns "$CC_MAX_TURNS" --timeout "$CC_TIMEOUT"
    [[ -f "$RECIPE" ]] || { echo "no recipe.json at ${RECIPE}; stopping."; break; }

    DE_BASE=$(python3 -c "import json;print(json.load(open('$RECIPE'))['data'].get('data_engine_base','pristine'))" 2>/dev/null || echo pristine)
    [[ "$DE_BASE" == "current" ]] || cp "$DATA_ENGINE_PRISTINE" "$DATA_ENGINE_WORK"

    DE_EDIT=$(cd "$RECIPE_DIR" 2>/dev/null && find revised_code -type f -name 'data_engine.py' 2>/dev/null | head -1)
    if [[ -n "$DE_EDIT" ]]; then
        cp "${RECIPE_DIR}/${DE_EDIT}" "$DATA_ENGINE_WORK"
        echo "  applied the recipe's data-engine edit"
    fi
    cp "$DATA_ENGINE_WORK" "${RECIPE_DIR}/data_engine_used.py"

    SHARED="${ENGINE_DIR}/iter${i}/data_engine_shared"
    ACTION=$(python3 -c "import json;print(json.load(open('$RECIPE'))['data'].get('action',''))")
    if [[ "$ACTION" == "reuse" ]]; then
        REUSE_FROM=$(python3 -c "import json;print(json.load(open('$RECIPE'))['data'].get('reuse_from',''))")
        SRC="${ENGINE_DIR}/${REUSE_FROM}/data_engine_shared"
        [[ -d "$SRC" ]] || { echo "reuse_from=${REUSE_FROM} missing"; break; }
        echo "=== (2) reuse the certified data from ${REUSE_FROM} ==="
        mkdir -p "$SHARED"; cp -a "${SRC}/." "${SHARED}/"
        DATA_SOURCE="reuse:${REUSE_FROM}"
    else
        echo "=== (2) regenerate data with the data engine (postprocessing applied inside) ==="
        OV_MIN=$(python3 -c "import json;print(json.load(open('$RECIPE'))['data'].get('data_engine_overrides',{}).get('min_target_examples',''))")
        OV_FILT=$(python3 -c "import json;print(json.load(open('$RECIPE'))['data'].get('data_engine_overrides',{}).get('episode_filter_top_pct',''))")
        conda activate "$EVAL_CONDA_ENV"
        python "$DATA_ENGINE_WORK" \
            --env "$ENV" \
            --codebase-dir "$(pwd)/$CODEBASE_DATA" \
            --results-dir "$(pwd)/$RESULTS_DIR" \
            --memory-dir "$(pwd)/$MEMORY_DIR" \
            --output-dir "$(pwd)/$SHARED" \
            --episode-filter-top-pct "${OV_FILT:-100}" \
            --min-target-examples "${OV_MIN:-300}" \
            --model "$CC_MODEL" --effort "$CC_EFFORT" \
            --max-turns "$CC_MAX_TURNS" --timeout "$CC_TIMEOUT"
        DATA_SOURCE="regenerate:iter${i}"
    fi
    SELECTED="${SHARED}/output/selected_data.json"
    [[ -f "$SELECTED" ]] || { echo "no selected_data.json; stopping."; break; }

    echo "=== (3) train + eval each recipe config ==="
    mapfile -t CONFIGS < <(python3 -c "import json;[print(c) for c in json.load(open('$RECIPE'))['train_configs']]")
    k=0
    for cfg in "${CONFIGS[@]}"; do
        [[ -z "$cfg" ]] && continue
        k=$((k+1)); [[ $k -gt $MAX_CONFIGS_PER_ITER ]] && break
        run_config "$i" "$DATA_SOURCE" "$cfg" "$SELECTED" "cfg${k}"
    done

    echo "=== iteration ${i} done; ledger: ${LEDGER} ==="
done

echo "=== engine complete. ledger: ${LEDGER} ==="
