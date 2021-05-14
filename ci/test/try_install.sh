#!/bin/bash
set -xe

build_dir=${ONEFLOW_BUILD_DIR:-"$PWD"}

if [ ! -z "$ONEFLOW_WHEEL_PATH" ] && [ compgen -G "${wheel_path}/*.whl" > /dev/null ]; then
    ls -la $ONEFLOW_WHEEL_PATH
    python3 -m pip install --user $ONEFLOW_WHEEL_PATH/*.whl
elif [ ! -z "$ONEFLOW_WHEEL_PATH" ] && [ -f "$ONEFLOW_WHEEL_PATH" ]; then
    python3 -m pip install --user "$ONEFLOW_WHEEL_PATH"
elif [ -d "$build_dir" ]; then
    source "${build_dir}/source.sh"
else
    echo "wheel not found: $ONEFLOW_WHEEL_PATH, src dir not found: $src_dir, continue anyway..."
fi
