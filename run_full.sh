#!/usr/bin/env bash

set -ux

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

MODEL_NAME=unet
PREFIX=$MODEL_NAME-n3

if [[ $MODEL_NAME == resnet50 ]]
then
  array=( 2600 2700 2800 2900 3000 3100 3200 3300 3400 3500 4000 5000 6000 7000 8000 9000 9900 )
  BS=115
elif [[ $MODEL_NAME == unet ]]
then
  array=( 3600 3700 3800 3900 4000 4500 5000 6000 7000 8000 9000 )
  BS=5
elif [[ $MODEL_NAME == densenet121 ]]
then
  array=( 1300 1350 1400 1450 1500 1550 1600 1700 1800 2000 2500 3000 4000 5000 6000 7000 8000 9000 9900 )
  BS=70
elif [[ $MODEL_NAME == swin_transformer ]]
then
  array=( 2300 2400 2500 2700 3000 3500 4000 4500 5000 6000 7000 8000 9000 9900 )
  BS=40
fi

ITERS=1

METHOD=ours
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --nlr --group-num 2
done

METHOD=ours-g4
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --nlr --group-num 4
done

METHOD=ours-g6
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --nlr --group-num 6
done

METHOD=me-style-mul-beta
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --nlr --me-style --me-method eq_mul_beta --group-num 2
done

METHOD=me-style-div-beta
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --nlr --me-style --me-method eq_div_beta --group-num 2
done

METHOD=ours-id-guided
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --group-num 2
done

METHOD=ours-g4-id-guided
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --group-num 4
done

METHOD=ours-g6-id-guided
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --group-num 6
done

METHOD=no-gp
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --no-lr --group-num 1
done

# ------

# METHOD=me-style
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --nlr --group-num 2 --me-style
# done
#
# METHOD=no-fbip
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --old-immutable --nlr --group-num 2
# done
#
# METHOD=raw-gp
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --group-num 2
# done
#
# METHOD=ours-with-size
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --nlr --group-num 2 --with-size
# done

# --------------

# METHOD=no-gp-size
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --no-lr --with-size --group-num 1
# done
#
# METHOD=no-fbip-size
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --old-immutable --nlr --group-num 2 --with-size
# done
#
# METHOD=raw-gp-size
# for threshold in "${array[@]}"
# do
#   ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --with-size --group-num 2
# done

METHOD=raw-dtr
for threshold in "${array[@]}"
do
  ONEFLOW_DTR_SUMMARY_FILE_PREFIX=$PREFIX-$METHOD-$threshold ONEFLOW_DISABLE_VIEW=1 python3 -u $SCRIPT_DIR/run_dtr.py $MODEL_NAME $BS ${threshold}mb $ITERS --no-dataloader --no-o-one --no-allo --group-num 1
done
