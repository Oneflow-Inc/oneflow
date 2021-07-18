#!/usr/bin/env bash

set -uex

cd $ONEFLOW_MODEL_DIR

function check_relative_speed {
  awk -F'[:(]' -v threshold=$1 'BEGIN { ret=2 } /Relative speed/{ if ($2 > threshold) { ret=0 } else { ret=1 }} {print $0} END { exit ret }'
}
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 16x3x224x224 | check_relative_speed 1.05
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 8x3x224x224 | check_relative_speed 1.05
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 4x3x224x224 | check_relative_speed 1
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 2x3x224x224 | check_relative_speed 0.8
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 1x3x224x224 | check_relative_speed 0.8
