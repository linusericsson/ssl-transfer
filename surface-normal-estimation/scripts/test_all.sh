#!/bin/bash

for MODEL_NAME in simclr-v1 simclr-v2 moco-v1 moco-v2 byol swav deepcluster-v2 sela-v2 infomin insdis pirl pcl-v1 pcl-v2 supervised supervised-simclr
do
    ./scripts/test_models.sh $MODEL_NAME | tee ../results/$MODEL_NAME.log
done
