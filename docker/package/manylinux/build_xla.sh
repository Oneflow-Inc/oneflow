set -ex
wheelhouse_dir=/oneflow-src/wheelhouse-xla
ONEFLOW_SRC_DIR=${ONEFLOW_SRC_DIR:-/oneflow-src}

# TF requires py3 to build
PY_ROOT=/opt/python/cp37-cp37m
PY_BIN=${PY_ROOT}/bin
export PATH=$PY_BIN:$PATH
python --version

source scl_source enable devtoolset-7

bash docker/package/manylinux/build_wheel.sh \
    --python3.7 \
    --cache-dir $ONEFLOW_SRC_DIR/manylinux2014-build-cache-cuda-10.2-xla \
    --house-dir $wheelhouse_dir \
    -DWITH_XLA=ON
