#!/bin/bash

FILENAME="eval_generic.py"
ARCHNAME_ENCODER="ptresnet50dilated"
ARCHNAME_DECODER="ppm"
DATASET="nyuv2sn40"
SPLIT="test"
IMAGE_MODE=${2:-rgb}
# folder where the checkpoints are stored
CHECKPOINT_FOLDER="/raid/hgouk/ssl-eval/finetuned/$1/baseline-ptresnet50dilated-ppm-ngpus1-batchSize2-imgMaxSize1000-paddingConst8-segmDownsampleRate8-LR_encoder0.02-LR_decoder0.02-epoch150"

# single gpu testing
GPU=0

SUFFIX="_latest.pth"
python3 "$FILENAME" \
     --arch_encoder "$ARCHNAME_ENCODER" \
     --arch_decoder "$ARCHNAME_DECODER" \
     --image_mode "$IMAGE_MODE" \
     --dataset "${DATASET}" \
     --split_name "$SPLIT" \
     --dirname "$CHECKPOINT_FOLDER" \
     --suffix "$SUFFIX" \
     --gpu $GPU
