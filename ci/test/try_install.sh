#!/bin/bash
set -xe

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
wheel_path=${ONEFLOW_WHEEL_PATH:-"$PWD/ci_tmp/oneflow-0.0.1-cp36-cp36m-linux_x86_64.whl"}

if [ -f "$wheel_path" ]; then
    pip3 install --user "$wheel_path"
elif [ -d "$src_dir" ]; then
    pip3 install -e "$src_dir" --user
else
    echo "wheel not found: $wheel_path, src dir not found: $src_dir, continue anyway..."
fi
