#!/usr/bin/env bash

set -x
set -e

export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH

EXTRA_ONEFLOW_CMAKE_ARGS=""
PY_VERS=()

while [[ "$#" > 0 ]]; do
    case $1 in
        --skip-third-party) SKIP_THIRD_PARTY=1; ;;
        --skip-wheel) SKIP_WHEEL=1; ;;
        --cache-dir) CACHE_DIR=$2; shift ;;
        --house-dir) HOUSE_DIR=$2; shift ;;
        --package-name) PACKAGE_NAME=$2; shift ;;
        --python3.5) PY_VERS+=( "35" ) ;;
        --python3.6) PY_VERS+=( "36" ) ;;
        --python3.7) PY_VERS+=( "37" ) ;;
        --python3.8) PY_VERS+=( "38" ) ;;
        *) EXTRA_ONEFLOW_CMAKE_ARGS="${EXTRA_ONEFLOW_CMAKE_ARGS} $1" ;;
    esac;
    shift;
done

if [[ ! -v CACHE_DIR ]]
then
    CACHE_DIR=$PWD/manylinux2014-build-cache
fi

if [[ ! -v HOUSE_DIR ]]
then
    HOUSE_DIR=$PWD/wheelhouse
fi

if [[ ! -v PACKAGE_NAME ]]
then
    PACKAGE_NAME=oneflow
fi

ONEFLOW_SRC_DIR=`cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd`
ONEFLOW_SRC_DIR=$ONEFLOW_SRC_DIR/../../..

if [[ ${#PY_VERS[@]} -eq 0 ]]
then
    PY_VERS=( 35 36 37 38 )
fi

cd $ONEFLOW_SRC_DIR

# TF requires py3 to build
export PATH=/opt/python/cp37-cp37m/bin:$PATH
python --version
gcc --version

# specify a mounted dir as bazel cache dir
export TEST_TMPDIR=$CACHE_DIR/bazel_cache

THIRD_PARTY_BUILD_DIR=$CACHE_DIR/build-third-party
THIRD_PARTY_INSTALL_DIR=$CACHE_DIR/build-third-party-install
COMMON_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DBUILD_RDMA=ON -DTHIRD_PARTY_DIR=$THIRD_PARTY_INSTALL_DIR"
if [[ $SKIP_THIRD_PARTY != 1 ]]; then
    mkdir -p $THIRD_PARTY_BUILD_DIR
    pushd $THIRD_PARTY_BUILD_DIR

    cmake -DTHIRD_PARTY=ON \
        $COMMON_CMAKE_ARGS \
        -DONEFLOW=OFF \
        $EXTRA_ONEFLOW_CMAKE_ARGS \
        $ONEFLOW_SRC_DIR
    make -j`nproc` prepare_oneflow_third_party

    popd
fi

ONEFLOW_BUILD_DIR=$CACHE_DIR/build-oneflow

function cleanup {
  set -x
  rm -rf $ONEFLOW_BUILD_DIR/python_scripts/oneflow/*.so
  rm -rf build/bdist.linux-x86_64
  rm -rf build/lib
  rm -rf tmp_wheel
}

for PY_VER in ${PY_VERS[@]}
do
    mkdir -p $ONEFLOW_BUILD_DIR
    pushd $ONEFLOW_BUILD_DIR
    PY_ABI=cp${PY_VER}-cp${PY_VER}
    if [[ $PY_VER != 38 ]]
    then
        PY_ABI=${PY_ABI}m
    fi
    PY_ROOT=/opt/python/${PY_ABI}
    PY_BIN=${PY_ROOT}/bin/python
    cleanup
    cmake -DTHIRD_PARTY=OFF -DONEFLOW=ON \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        $COMMON_CMAKE_ARGS \
        -DPython3_EXECUTABLE=${PY_BIN} \
        $EXTRA_ONEFLOW_CMAKE_ARGS \
        $ONEFLOW_SRC_DIR
    cmake --build . -j `nproc`
    popd
    trap cleanup EXIT
    if [[ $SKIP_WHEEL != 1 ]]; then
        rm -rf $ONEFLOW_BUILD_DIR/python_scripts/*.egg-info
        $PY_BIN setup.py bdist_wheel -d tmp_wheel --build_dir $ONEFLOW_BUILD_DIR --package_name $PACKAGE_NAME
        auditwheel repair tmp_wheel/*.whl --wheel-dir $HOUSE_DIR
    fi
    cleanup
done
