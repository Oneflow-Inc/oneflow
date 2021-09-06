source scl_source enable devtoolset-7
set -ex
gcc --version
export PATH=/usr/lib64/ccache:$PATH
mkdir -p ${ONEFLOW_CI_BUILD_DIR}
cd ${ONEFLOW_CI_BUILD_DIR}
find ${ONEFLOW_CI_BUILD_DIR} -name CMakeCache.txt -delete
cmake -S ${ONEFLOW_CI_SRC_DIR} -C ${ONEFLOW_CI_CMAKE_INIT_CACHE} -DPython3_EXECUTABLE=${ONEFLOW_CI_PYTHON_EXE}
cmake --build . -j $(nproc)
cd ${ONEFLOW_CI_SRC_DIR}
cd python
python setup.py bdist_wheel
