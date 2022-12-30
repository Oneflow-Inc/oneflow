set -ex
export PATH=/usr/lib/llvm-15/bin:/usr/lib64/ccache:/root/.local/bin:$PATH

# clean python dir
cd ${ONEFLOW_CI_SRC_DIR}
${ONEFLOW_CI_PYTHON_EXE} -m pip install -i https://mirrors.aliyun.com/pypi/simple --user -r ci/fixed-dev-requirements.txt
cd python
git config --global --add safe.directory ${ONEFLOW_CI_SRC_DIR}
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

# build pip
cd ${ONEFLOW_CI_SRC_DIR}
cd python
${ONEFLOW_CI_PYTHON_EXE} setup.py bdist_wheel
