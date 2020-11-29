#!/bin/bash
conda activate /data/saandeepaath/my_envs/.flow
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 /data/saandeepaath/flow_based/main.py --root /data/saandeepaath/flow_based/data/
conda deactivate