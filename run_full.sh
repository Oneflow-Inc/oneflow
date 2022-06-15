#!/usr/bin/env bash

set -ux

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export ONEFLOW_DTR_MODEL_NAME=unet-2

array=( 4000 5000 6000 7000 8000 9000 )
BS=5

# METHOD=nlr-high-conv
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$ONEFLOW_DTR_MODEL_NAME-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py $BS ${threshold}mb 40 tmp --no-dataloader --no-o-one --nlr --high-conv
# done

METHOD=nlr
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$ONEFLOW_DTR_MODEL_NAME-$METHOD-$threshold CUDA_VISIBLE_DEVICES=1 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py $BS ${threshold}mb 40 tmp --no-dataloader --no-o-one --nlr
done

METHOD=no-lr
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$ONEFLOW_DTR_MODEL_NAME-$METHOD-$threshold CUDA_VISIBLE_DEVICES=1 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py $BS ${threshold}mb 40 tmp --no-dataloader --no-o-one --no-lr
done

METHOD=no-allo
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$ONEFLOW_DTR_MODEL_NAME-$METHOD-$threshold CUDA_VISIBLE_DEVICES=1 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py $BS ${threshold}mb 40 tmp --no-dataloader --no-o-one --no-allo
done

METHOD=normal
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$ONEFLOW_DTR_MODEL_NAME-$METHOD-$threshold CUDA_VISIBLE_DEVICES=1 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py $BS ${threshold}mb 40 tmp --no-dataloader --no-o-one
done
