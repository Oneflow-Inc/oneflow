#!/usr/bin/env bash

set -uex

cd $ONEFLOW_MODEL_DIR
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 8x3x224x224
