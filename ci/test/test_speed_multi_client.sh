#!/usr/bin/env bash

set -uxo pipefail

rc=0
trap 'rc=$?' ERR

cd $ONEFLOW_MODELS_DIR

function check_relative_speed {
  awk -F'[:(]' -v threshold=$1 'BEGIN { ret=2 } /Relative speed/{ if ($2 > threshold) { printf "✔️ "; ret=0 } else { printf "❌ "; ret=1 }} {print $0} END { exit ret }'
}

function write_to_file_and_print {
  tee -a result
  printf "\n" >> result
}

python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 16x3x224x224 --no-show-memory --times 50 | check_relative_speed 1.05 | write_to_file_and_print
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 8x3x224x224 --no-show-memory --times 50 | check_relative_speed 1.05 | write_to_file_and_print
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 4x3x224x224 --no-show-memory --times 50 | check_relative_speed 1.0 | write_to_file_and_print
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 2x3x224x224 --no-show-memory --times 50 | check_relative_speed 0.9 | write_to_file_and_print
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 1x3x224x224 --no-show-memory --times 50 | check_relative_speed 0.9 | write_to_file_and_print

python3 -m oneflow.distributed.launch --nproc_per_node 2 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 16x3x224x224 --no-show-memory --times 50 --ddp | check_relative_speed 1.0 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 8x3x224x224 --no-show-memory --times 50 --ddp | check_relative_speed 0.95 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 4x3x224x224 --no-show-memory --times 50 --ddp | check_relative_speed 0.95 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 2x3x224x224 --no-show-memory --times 50 --ddp | check_relative_speed 0.92 | write_to_file_and_print
python3 -m oneflow.distributed.launch --nproc_per_node 2 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 1x3x224x224 --no-show-memory --times 50 --ddp | check_relative_speed 0.85 | write_to_file_and_print

result="GPU Name: `nvidia-smi --query-gpu=name --format=csv,noheader -i 0` \n\n `cat result`"
# escape newline for github actions: https://github.community/t/set-output-truncates-multiline-strings/16852/2
# note that we escape \n and \r to \\n and \\r (i.e. raw string "\n" and "\r") instead of %0A and %0D, 
# so that they can be correctly handled in javascript code
result="${result//'%'/'%25'}"
result="${result//$'\n'/'\\n'}"
result="${result//$'\r'/'\\r'}"

echo "::set-output name=stats::$result"

exit $rc
