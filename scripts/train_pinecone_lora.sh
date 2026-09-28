#!/usr/bin/env bash
set -euo pipefail

# Reproducible single-GPU LoRA launcher for Eagle/LocateAnything.
# Set META_PATH to the generated recipe JSON before running. Every uppercase
# setting can be overridden from the environment.

REPO_DIR="${REPO_DIR:-${HOME}/projects/Eagle/Embodied}"
VENV_DIR="${VENV_DIR:-${HOME}/venvs/locateanything}"
MODEL_PATH="${MODEL_PATH:-${HOME}/model-cache/LocateAnything-3B}"
META_PATH="${META_PATH:?Set META_PATH to the LocateAnything recipe JSON}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/work_dirs/locateanything_pinecone/lora_r32}"

MAX_STEPS="${MAX_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
GRADIENT_ACC="${GRADIENT_ACC:-4}"
LORA_RANK="${LORA_RANK:-32}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-4096}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
MASTER_PORT="${MASTER_PORT:-29517}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${VENV_DIR}/bin:${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}:${MODEL_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM="true"
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export DIST_BACKEND="${DIST_BACKEND:-gloo}"

test -x "${VENV_DIR}/bin/python"
test -d "${REPO_DIR}"
test -d "${MODEL_PATH}"
test -f "${META_PATH}"
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"
LAUNCHER=pytorch python -m torch.distributed.run \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  eaglevl/train/locany_finetune_magi_stream.py \
  --model_name_or_path "${MODEL_PATH}" \
  --meta_path "${META_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --overwrite_output_dir False \
  --block_size 6 \
  --attn_implementation sdpa \
  --causal_attn False \
  --freeze_llm True \
  --freeze_backbone True \
  --freeze_mlp True \
  --use_llm_lora "${LORA_RANK}" \
  --use_backbone_lora 0 \
  --unfreeze_vit_layers 0 \
  --vision_select_layer -1 \
  --dataloader_num_workers "${DATALOADER_WORKERS}" \
  --bf16 True \
  --tf32 True \
  --num_train_epochs 1 \
  --max_steps "${MAX_STEPS}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "${GRADIENT_ACC}" \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 3 \
  --learning_rate "${LEARNING_RATE}" \
  --weight_decay 0.01 \
  --warmup_steps "${WARMUP_STEPS}" \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --logging_steps 1 \
  --packing_buffer_size "${PACKING_BUFFER_SIZE}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --max_num_tokens_per_sample "${MAX_SEQ_LENGTH}" \
  --max_num_tokens "${MAX_NUM_TOKENS}" \
  --sample_log_interval 10 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --save_every_n_hours 0 \
  --use_onelogger False \
  --report_to none \
  --seed 42 \
  --data_seed 42 \
  --mlp_connector_layers 2 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
