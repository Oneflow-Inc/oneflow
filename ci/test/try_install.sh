#!/bin/bash
set -xe

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
wheel_path=${ONEFLOW_WHEEL_PATH:-"ci_tmp/*.whl"}

if [ -f "$wheel_path" ]; then
    pip3 install --user "$wheel_path"
elif [ -d "$src_dir" ]; then
    pip3 install -e "$src_dir" --user
else
    echo "wheel not found: $wheel_path, src dir not found: $src_dir, continue anyway..."
fi
