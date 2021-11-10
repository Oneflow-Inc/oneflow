#!/usr/bin/env bash

set -ex


cd $ONEFLOW_MODELS_DIR
git checkout test_resnet50_with_ci

python3 -m oneflow.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 Vision/classification/image/resnet50/train.py --ofrecord-path /dataset/imagenette/ofrecord --ofrecord-part-num 1 --num-devices-per-node 1 --lr 0.004 --momentum 0.875 --num-epochs 1 --train-batch-size 4 --val-batch-size 50 --print-interval 10 --exit-num 1 --use-fp16 --scale-grad --graph --use-gpu-decode

python3 -m oneflow.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 Vision/classification/image/resnet50/train.py --ofrecord-path /dataset/imagenette/ofrecord --ofrecord-part-num 1 --num-devices-per-node 1 --lr 0.004 --momentum 0.875 --num-epochs 1 --train-batch-size 4 --val-batch-size 50 --print-interval 10 --exit-num 1 --ddp

