set -ex
ONEFLOW_SRC_DIR=${ONEFLOW_SRC_DIR:-${PWD}}
wheelhouse_dir=${ONEFLOW_SRC_DIR}/wheelhouse-xla

# TF requires py3 to build
PY_ROOT=/opt/python/cp37-cp37m
PY_BIN=${PY_ROOT}/bin
export PATH=$PY_BIN:$PATH
python --version

source scl_source enable devtoolset-7

cache_dir=$ONEFLOW_SRC_DIR/manylinux2014-build-cache-cuda-10.2-xla
cache_dir=$ONEFLOW_SRC_DIR/manylinux2014-build-cache-cuda-11.0-xla
export TEST_TMPDIR=$cache_dir/bazel_cache
gcc --version

bash docker/package/manylinux/build_wheel.sh \
    --python3.6 \
    --cache-dir $cache_dir \
    --house-dir $wheelhouse_dir \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DWITH_XLA=ON
