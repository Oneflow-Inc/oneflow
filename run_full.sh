#!/usr/bin/env bash

set -ux

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

MODEL_NAME=densenet121
PREFIX=$MODEL_NAME-6

if [[ $MODEL_NAME == resnet50 ]]
then
  array=( 3000 3200 4000 5000 6000 7000 8000 9000 10000 )
  BS=115
elif [[ $MODEL_NAME == unet ]]
then
  array=( 4000 5000 6000 7000 8000 9000 )
  BS=5
elif [[ $MODEL_NAME == densenet121 ]]
then
  array=( 1700 2000 3000 4000 5000 6000 7000 8000 9000 10000 )
  BS=70
fi

METHOD=ours
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb 40 --no-dataloader --no-o-one --nlr --with-size
done
#
# METHOD=me-style
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb 40 --no-dataloader --no-o-one --nlr --me-style
# done
#
# METHOD=ours-without-size
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb 40 --no-dataloader --no-o-one --nlr
# done
#
# METHOD=no-gp
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb 40 --no-dataloader --no-o-one --no-lr --with-size
# done
#
# METHOD=raw-dtr
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb 40 --no-dataloader --no-o-one --no-allo
# done
#
# METHOD=no-fbip
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb 40 --no-dataloader --no-o-one --old-immutable --nlr --with-size
# done
#
# METHOD=raw-gp
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb 40 --no-dataloader --no-o-one --with-size
# done
