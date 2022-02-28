#!/usr/bin/env bash

set -uxo pipefail

rc=0
trap 'rc=$?' ERR

cd $ONEFLOW_MODELS_DIR

function check_relative_speed {
  awk -F'[:(]' -v threshold=$1 'BEGIN { ret=2 } /Relative speed/{ if ($2 >= threshold) { printf "✔️ "; ret=0 } else { printf "❌ "; ret=1 }} {print $0} END { exit ret }'
}

function check_millisecond_time {
  awk -F'[:(]' -v threshold=$1 'BEGIN { ret=2 } /OneFlow/{ if (substr($2, 2, length($2) - 4) <= threshold) { printf "✔️ "; ret=0 } else { printf "❌ "; ret=1 }} { print $0 } END { exit ret }'
}

function write_to_file_and_print {
  tee -a result
  printf "\n" >> result
}

python3 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 16x3x224x224 --no-show-memory --times 100 | check_relative_speed 1.01 | check_millisecond_time 137 | write_to_file_and_print
python3 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 8x3x224x224 --no-show-memory --times 100 | check_relative_speed 1.02 | check_millisecond_time 78.8 | write_to_file_and_print
python3 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 4x3x224x224 --no-show-memory --times 200 | check_relative_speed 0.99 | write_to_file_and_print
python3 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 2x3x224x224 --no-show-memory --times 200 | check_relative_speed 0.99 | write_to_file_and_print
python3 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 1x3x224x224 --no-show-memory --times 200 | check_relative_speed 0.92 | write_to_file_and_print

python3 -m oneflow.distributed.launch --nproc_per_node 2 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 16x3x224x224 --no-show-memory --times 100 --ddp | check_relative_speed 1.03 | check_millisecond_time 150 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 8x3x224x224 --no-show-memory --times 100 --ddp | check_relative_speed 0.99 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 4x3x224x224 --no-show-memory --times 200 --ddp | check_relative_speed 0.91 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 2x3x224x224 --no-show-memory --times 200 --ddp | check_relative_speed 0.83 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 scripts/compare_speed_with_pytorch.py Vision/classification/image/resnet50/models/resnet50.py resnet50 1x3x224x224 --no-show-memory --times 200 --ddp | check_relative_speed 0.80 | write_to_file_and_print

result="GPU Name: `nvidia-smi --query-gpu=name --format=csv,noheader -i 0` \n\n `cat result`"
# escape newline for github actions: https://github.community/t/set-output-truncates-multiline-strings/16852/2
# note that we escape \n and \r to \\n and \\r (i.e. raw string "\n" and "\r") instead of %0A and %0D, 
# so that they can be correctly handled in javascript code
result="${result//'%'/'%25'}"
result="${result//$'\n'/'\\n'}"
result="${result//$'\r'/'\\r'}"

echo "::set-output name=stats::$result"

exit $rc
