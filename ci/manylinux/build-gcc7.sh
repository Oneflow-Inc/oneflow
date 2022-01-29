source scl_source enable devtoolset-7
set -ex
ONEFLOW_CI_BUILD_PARALLEL=${ONEFLOW_CI_BUILD_PARALLEL:-$(nproc)}
gcc --version
ld --version
# clean python dir
cd ${ONEFLOW_CI_SRC_DIR}
${ONEFLOW_CI_PYTHON_EXE} -m pip install -i https://mirrors.aliyun.com/pypi/simple --user -r ci/fixed-dev-requirements.txt
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
export PATH="${PATH}:$(dirname ${ONEFLOW_CI_PYTHON_EXE})"
export PYTHON_BIN_PATH=${ONEFLOW_CI_PYTHON_EXE}
cmake -S ${ONEFLOW_CI_SRC_DIR} -C ${ONEFLOW_CI_CMAKE_INIT_CACHE} -DPython3_EXECUTABLE=${ONEFLOW_CI_PYTHON_EXE}

# cmake build
cd ${ONEFLOW_CI_BUILD_DIR}
cmake --build . --parallel ${ONEFLOW_CI_BUILD_PARALLEL}

# build pip
cd ${ONEFLOW_CI_SRC_DIR}
cd python
${ONEFLOW_CI_PYTHON_EXE} setup.py bdist_wheel
