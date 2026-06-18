#!/bin/bash

ADAPTATION="finetune"
MODEL="RETFound_mae"
MODEL_ARCH="retfound_mae"
FINETUNE="weights/RETFound_oct_weights.pth"   # وزن تو

DATA_PATH="oimhs_data"
NUM_CLASS=4
BATCH_SIZE=8          # اگر GPU memory ارور داد، به 4 تغییر بده

torchrun --nproc_per_node=1 --master_port=48766 main_finetune.py \
  --model "${MODEL}" \
  --model_arch "${MODEL_ARCH}" \
  --finetune "${FINETUNE}" \
  --savemodel \
  --global_pool \
  --batch_size ${BATCH_SIZE} \
  --world_size 1 \
  --epochs 30 \
  --nb_classes "${NUM_CLASS}" \
  --data_path "${DATA_PATH}" \
  --input_size 224 \
  --task "OIMHS_finetune" \
  --adaptation "${ADAPTATION}"