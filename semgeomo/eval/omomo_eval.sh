#!/bin/bash

python3 -m eval.eval_controlmdm \
    --model_path "/storage/group/4dvlab/wangzy/SemGeoMo/exp/omomo-stage2/model000080000.pt" \
    --dataset omomo \
    --replication_times 10 \
    --mask_ratio 1 \
    --bfgs_times_first 5 \
    --control_joint ["left_wrist","right_wrist"] \
    --bfgs_times_last 10 \
    --bfgs_interval 1 \
    --checkpoint="/storage/group/4dvlab/wangzy/SemGeoMo/exp/omomo-stage1/weights/model-26.pt" \
    --use_posterior 

