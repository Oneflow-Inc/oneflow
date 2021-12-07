set -ex
export ONEFLOW_CI_LLVM_DIR=/usr/lib/llvm-12
export PATH=$ONEFLOW_CI_LLVM_DIR/bin:/usr/lib64/ccache:/root/.local/bin:$PATH
export CC=$ONEFLOW_CI_LLVM_DIR/bin/clang
export CXX=$ONEFLOW_CI_LLVM_DIR/bin/clang++

# clean python dir
cd ${ONEFLOW_CI_SRC_DIR}
${ONEFLOW_CI_PYTHON_EXE} -m pip install -i https://mirrors.aliyun.com/pypi/simple --user -r ci/fixed-dev-requirements.txt
${ONEFLOW_CI_PYTHON_EXE} -m pip install -i https://mirrors.aliyun.com/pypi/simple --user cmake==3.22.0 ninja==1.10.2.3
cd python
git clean -nXd -e \!dist -e \!dist/**
git clean -fXd -e \!dist -e \!dist/**

# cmake config
mkdir -p ${ONEFLOW_CI_BUILD_DIR}
cd ${ONEFLOW_CI_BUILD_DIR}
find ${ONEFLOW_CI_BUILD_DIR} -name CMakeCache.txt
find ${ONEFLOW_CI_BUILD_DIR} -name CMakeCache.txt -delete
if [ ! -f "$ONEFLOW_CI_CMAKE_INIT_CACHE" ]; then
    echo "$ONEFLOW_CI_CMAKE_INIT_CACHE does not exist."
    exit 1
fi
cmake -S ${ONEFLOW_CI_SRC_DIR} -C ${ONEFLOW_CI_CMAKE_INIT_CACHE} -DPython3_EXECUTABLE=${ONEFLOW_CI_PYTHON_EXE}
# cmake build
cd ${ONEFLOW_CI_BUILD_DIR}
cmake --build . -j $(nproc)
