set -ex
wheelhouse_dir=/oneflow-src/wheelhouse-xla
ONEFLOW_SRC_DIR=${ONEFLOW_SRC_DIR:-${PWD}}

# TF requires py3 to build
PY_ROOT=/opt/python/cp37-cp37m
PY_BIN=${PY_ROOT}/bin
export PATH=$PY_BIN:$PATH
python --version

source scl_source enable devtoolset-7

cache_dir=$ONEFLOW_SRC_DIR/manylinux2014-build-cache-cuda-10.2-xla
export TEST_TMPDIR=$cache_dir/bazel_cache
bash docker/package/manylinux/build_wheel.sh \
    --python3.7 \
    --cache-dir $cache_dir \
    --house-dir $wheelhouse_dir \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DWITH_XLA=ON
