#!/usr/bin/env bash

MGPU=1
FILENAME="train_generic.py"

OUTPUT_FOLDER="/raid/hgouk/ssl-eval/finetuned/$1"
mkdir -p $OUTPUT_FOLDER
IMAGE_MODE=${2:-rgb}
# path to the pretrained weights
# PRETRAINED_MODEL="ae_state_dict.pth"
PRETRAINED_MODEL=/raid/hgouk/ssl-eval/models/$1.pth
# Number of epochs is set to 400 so that the learning rate decay scheme does not need to be modified.
# The model converges within the first 150 epochs.

python3 "$FILENAME" \
  --arch_encoder "ptresnet50dilated" \
  --arch_decoder "ppm" \
  --gpus "0" \
  --image_mode "$IMAGE_MODE" \
  --weights_encoder "$PRETRAINED_MODEL" \
  --dataset "nyuv2sn40" \
  --random_flip 0 \
  --freeze_until "layer4" \
  --ckpt "$OUTPUT_FOLDER"  \
  --epoch_iters 90 \
  --num_epoch 150
