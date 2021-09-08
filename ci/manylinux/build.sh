set -ex
export PATH=/usr/lib64/ccache:$PATH

# clean python dir
cd ${ONEFLOW_CI_SRC_DIR}
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
fi
cmake -S ${ONEFLOW_CI_SRC_DIR} -C ${ONEFLOW_CI_CMAKE_INIT_CACHE} -DPython3_EXECUTABLE=${ONEFLOW_CI_PYTHON_EXE}

# cmake build
cd ${ONEFLOW_CI_BUILD_DIR}
cmake --build . -j $(nproc)

# build pip
cd ${ONEFLOW_CI_SRC_DIR}
cd python
${ONEFLOW_CI_PYTHON_EXE} setup.py bdist_wheel
