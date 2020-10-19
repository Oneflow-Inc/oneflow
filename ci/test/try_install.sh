#!/bin/bash
set -xe

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
wheel_path=${ONEFLOW_WHEEL_PATH:-"$PWD/wheelhouse"}

if [ -d "$wheel_path" ]; then
    ls -la $wheel_path
    pip3 install --user $wheel_path/*.whl
elif [ -e "$wheel_path" ]; then
    pip3 install --user "$wheel_path"
elif [ -d "$src_dir" ]; then
    pip3 install -e "$src_dir" --user
else
    echo "wheel not found: $wheel_path, src dir not found: $src_dir, continue anyway..."
fi
