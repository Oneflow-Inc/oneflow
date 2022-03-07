#!/usr/bin/env bash

set -ex

cd $ONEFLOW_MODELS_DIR

OFRECORD_PATH=/dataset/imagenette/ofrecord
if [ ! -d "/dataset/imagenette/ofrecord/train" ];then
    mkdir -p ./dataset/ofrecord
    ln -s /dataset/imagenette/ofrecord ./dataset/ofrecord/train
    OFRECORD_PATH=./dataset/ofrecord
fi

python3 -m oneflow.distributed.launch --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 Vision/classification/image/resnet50/train.py --ofrecord-path $OFRECORD_PATH --ofrecord-part-num 1 --num-devices-per-node 1 --lr 0.004 --momentum 0.875 --num-epochs 1 --train-batch-size 4 --val-batch-size 50 --print-interval 10 --exit-num 1 --ddp
python3 -m oneflow.distributed.launch --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr 127.0.0.1 Vision/classification/image/resnet50/train.py --ofrecord-path $OFRECORD_PATH --ofrecord-part-num 2 --num-devices-per-node 1 --lr 0.004 --momentum 0.875 --num-epochs 1 --train-batch-size 4 --val-batch-size 50 --print-interval 10 --exit-num 1 --use-fp16 --channel-last --scale-grad --graph --fuse-bn-relu --fuse-bn-add-relu --use-gpu-decode
