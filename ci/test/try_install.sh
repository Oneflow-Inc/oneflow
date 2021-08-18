#!/bin/bash
set -xe

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
wheel_path=${ONEFLOW_WHEEL_PATH:-"$PWD/wheelhouse"}
index=${ONEFLOW_PIP_INDEX}
pkg_name=${ONEFLOW_PACKAGE_NAME:-"oneflow"}

if [ -n "$index" ]; then
    python3 -m pip install --find-links ${index} ${pkg_name}
elif [ -d "$wheel_path" ]; then
    ls -la $wheel_path
    export PATH=/root/.local/bin:$PATH
    python3 -m pip install https://oneflow-static.oss-cn-beijing.aliyuncs.com/pipindex/pipindex-0.1.3-py2.py3-none-any.whl --user
    pipindex build $wheel_path
    python3 -m pip install -U --user --extra-index-url file://${wheel_path}/simple ${pkg_name}
elif [ -e "$wheel_path" ]; then
    python3 -m pip install --user "$wheel_path"
elif [ -d "$src_dir" ]; then
    python3 -m pip install -e "$src_dir" --user
else
    echo "wheel not found: $wheel_path, src dir not found: $src_dir, continue anyway..."
fi
