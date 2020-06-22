#!/usr/bin/env bash

set -x
set -e

EXTRA_ONEFLOW_CMAKE_ARGS=""
PY_VERS=()

while [[ "$#" > 0 ]]; do 
    case $1 in
        --skip-third-party) SKIP_THIRD_PARTY=1; ;;
        --python3.5) PY_VERS+=( "35" ) ;;
        --python3.6) PY_VERS+=( "36" ) ;;
        --python3.7) PY_VERS+=( "37" ) ;;
        --python3.8) PY_VERS+=( "38" ) ;;
        *) EXTRA_ONEFLOW_CMAKE_ARGS="${EXTRA_ONEFLOW_CMAKE_ARGS} $1" ;;
    esac;
    shift;
done

if [[ ${#PY_VERS[@]} -eq 0 ]]
then
    PY_VERS=( 35 36 37 38 )
fi

DIR=`cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd`
DIR=$DIR/../../..
cd $DIR

if [[ $SKIP_THIRD_PARTY != 1 ]]; then
    mkdir -p build-manylinux2014-third-party
    cd build-manylinux2014-third-party

    cmake -DTHIRD_PARTY=ON -DCMAKE_BUILD_TYPE=Release .. 
    make -j`nproc`

    cd ..
fi

export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
for PY_VER in ${PY_VERS[@]}
do
    BUILD_DIR=build-manylinux2014-py$PY_VER
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    PY_ABI=cp${PY_VER}-cp${PY_VER}
    if [[ $PY_VER != 38 ]]
    then
        PY_ABI=${PY_ABI}m
    fi
    PY_ROOT=/opt/python/${PY_ABI}
    PY_BIN=${PY_ROOT}/bin/python
    $PY_BIN -m pip install numpy protobuf
    cmake -DTHIRD_PARTY=OFF         \
        -DPython3_ROOT_DIR=$PY_ROOT \
        -DCMAKE_BUILD_TYPE=Release  \
        $EXTRA_ONEFLOW_CMAKE_ARGS   \
        ..
    cmake --build . -j `nproc`
    cd ..
    $PY_BIN setup.py bdist_wheel -d tmp_wheel --build_dir $BUILD_DIR
    auditwheel repair tmp_wheel/*.whl
    rm -rf tmp_wheel
done
