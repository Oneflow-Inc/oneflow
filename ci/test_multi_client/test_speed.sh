#!/usr/bin/env bash

set -ex

git clone https://github.com/Oneflow-Inc/models 
cd models
# TODO: remove this line when merged
git co add_speed_comparing_script
python3 scripts/compare_speed_with_pytorch.py resnet50/models/resnet50.py resnet50 8x3x224x224
