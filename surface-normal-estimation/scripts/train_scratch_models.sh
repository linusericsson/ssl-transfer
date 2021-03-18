#!/bin/bash

MGPU=0
FILENAME="train_generic.py"


OUTPUT_FOLDER="output/resnet50_scratch"
mkdir -p $OUTPUT_FOLDER


# Number of epochs is set to 2000 so that the learning rate decay scheme does not need to be modified.
# The model converges within the first 400 epochs.

python3 "$FILENAME" \
  --arch_encoder "ptresnet50dilated" \
  --arch_decoder "ppm" \
  --gpus "0-$MGPU" \
  --image_mode "rgb" \
  --dataset "nyuv2sn40" \
  --random_flip 0 \
  --ckpt "$OUTPUT_FOLDER" \
  --epoch_iters 50 \
  --num_epoch 2000
