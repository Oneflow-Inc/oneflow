#!/usr/bin/env bash

set -ux

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

MODEL_NAME=resnet50
PREFIX=$MODEL_NAME-4

array=( 3000 4000 5000 6000 7000 8000 9000 10000 )
BS=115

# METHOD=nlr-high-conv
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$MODEL_NAME-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $BS ${threshold}mb 40 --no-dataloader --no-o-one --nlr --high-conv
# done

METHOD=nlr
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb 40 --no-dataloader --no-o-one --nlr
done

METHOD=no-lr
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb 40 --no-dataloader --no-o-one --no-lr
done

METHOD=no-allo
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb 40 --no-dataloader --no-o-one --no-allo
done

METHOD=old-immutable
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb 40 --no-dataloader --no-o-one --old-immutable
done

METHOD=normal
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb 40 --no-dataloader --no-o-one
done
