#!/bin/bash

for d in "byol" "deepcluster-v2" "moco-v1" "moco-v2" "pirl" "sela-v2" "supervised" "swav"
do
	python eval_multipro.py --gpus 0,1 --cfg selfsupconfig/${d}.yaml | tee results/${d}.txt
done
