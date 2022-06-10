#!/usr/bin/env bash

set -ux

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export ONEFLOW_DTR_MODEL_NAME=resnet152

array=( 3000 4000 5000 6000 7000 8000 9000 10000 )

METHOD=nlr
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$ONEFLOW_DTR_MODEL_NAME-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py 55 ${threshold}mb 40 tmp --no-dataloader --no-o-one --nlr
done

# METHOD=no-lr
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$ONEFLOW_DTR_MODEL_NAME-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py 55 ${threshold}mb 40 tmp --no-dataloader --no-o-one --no-lr
# done
#
# METHOD=no-allo
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$ONEFLOW_DTR_MODEL_NAME-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py 55 ${threshold}mb 40 tmp --no-dataloader --no-o-one --no-allo
# done
#
# METHOD=normal
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$ONEFLOW_DTR_MODEL_NAME-$METHOD-$threshold CUDA_VISIBLE_DEVICES=3 ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/rn50_dtr.py 55 ${threshold}mb 40 tmp --no-dataloader --no-o-one
# done


# python3 $SCRIPT_DIR/run.py 10 115 4500mb 40 --no-o-one
# python3 $SCRIPT_DIR/run.py 10 115 4500mb 40 --no-o-one --high-add-n --high-conv --nlr
# python3 $SCRIPT_DIR/run.py 10 115 4000mb 40 --no-o-one --high-add-n --high-conv --nlr
# python3 $SCRIPT_DIR/run.py 10 115 3500mb 40 --no-o-one --high-add-n --high-conv --nlr
# python3 $SCRIPT_DIR/run.py 10 115 4500mb 40 --no-o-one --high-conv
# python3 $SCRIPT_DIR/run.py 10 115 4500mb 40 --no-o-one --high-conv --no-lr
# python3 $SCRIPT_DIR/run.py 1 115 4000mb 40 --no-o-one --no-allocator
