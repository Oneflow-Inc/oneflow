#!/usr/bin/env bash

set -uxo pipefail

rc=0
trap 'rc=$?' ERR

cd $ONEFLOW_MODEL_DIR

function check_relative_speed {
  awk -F'[:(]' -v threshold=$1 'BEGIN { ret=2 } /Relative speed/{ if ($2 > threshold) { ret=0 } else { ret=1 }} {print $0} END { exit ret }'
}

function write_to_file_and_print {
  tee -a result
  printf "\n" >> result
}

python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 16x3x224x224 | check_relative_speed 1.05 | write_to_file_and_print
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 8x3x224x224 | check_relative_speed 1.05 | write_to_file_and_print
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 4x3x224x224 | check_relative_speed 1 | write_to_file_and_print
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 2x3x224x224 | check_relative_speed 0.8 | write_to_file_and_print
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 1x3x224x224 | check_relative_speed 0.8 | write_to_file_and_print

# newline needs to be escaped: https://github.community/t/set-output-truncates-multiline-strings/16852/2
result=`cat result`
result="${result//'%'/'%25'}"
result="${result//$'\n'/'%0A'}"
result="${result//$'\r'/'%0D'}"

echo "::set-output name=stats::$result"

exit $rc
