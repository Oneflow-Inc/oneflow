# Oneflow
![Build and Test CI](https://github.com/Oneflow-Inc/oneflow/workflows/Build%20and%20Test%20CI/badge.svg?branch=develop)

### Build OneFlow from Source

  Building OneFlow from source requires a `BLAS libary` installed. On CentOS, if you have `Intel MKL` installed, please update the environment variable. 

  ```
  export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
  ```

  Or you can install OpenBLAS and other tools through:

  ```
  sudo yum -y install epel-release && sudo yum -y install git gcc-c++ cmake3 openblas-devel kernel-devel-$(uname -r) nasm
  ```

- #### Clone Source Code

  Clone source code and submodules (faster, recommended)

  ```
  git clone https://github.com/Oneflow-Inc/oneflow
  git submodule update --init --recursive
  ```

  or you can also clone the repo with `--recursive` flag to clone third_party submodules together

  ```
  git clone https://github.com/Oneflow-Inc/oneflow --recursive
  ```

- #### Enter Build Directory

  ```
  cd build
  ```

- #### Build Third Party from Source

  Inside directory `build`, run:
  ```
  cmake -DTHIRD_PARTY=ON .. && make -j
  ```

- #### Build OneFlow

  Inside directory `build`, run:
  ```
  cmake .. \
  -DTHIRD_PARTY=OFF \
  -DPython_NumPy_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
  -DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
  -DPYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])")

  make -j$(nproc)
  ```

- #### Install OneFlow

  In the root path of OneFlow repo, run:
  ```
  pip3 install -e . --user
  ```

  Alternatively, you can also install OneFlow by adding `build/python_scripts` to your `PYTHONPATH`

  For example:
  ```
  export PYTHONPATH=$HOME/oneflow/build/python_scripts:$PYTHONPATH
  ```

- #### Generate Pip package

  In the root path of OneFlow repo, run:
  ```
  python3 setup.py bdist_wheel
  ```
  Your should find a `.whl` package in `dist`.

### Build with XLA

- Install bazel

  Download and install bazel from [here](https://docs.bazel.build/versions/1.0.0/bazel-overview.html) , and version 0.24.1 is recommended. You can confirm bazel is installed successfully by running the following command:

  ```shell
  bazel version
  ```

- Update cmake

  It is needed only if CMake installed does not support downloading .tgz file from URL with https protocol. Skip this step, just go back here to reinstall CMake if you encountered a downloading error while building the third-parties.

  Download cmake(>=3.7) from [here](https://cmake.org/download/) , configure and install it by the following command:

  ```shell
  # Install curl develop toolkit
  sudo yum install libcurl-devel
 
  # install cmake
  cd cmake && ./bootstrap --system-curl --prefix=$your_path && make install
  ```

- Build third-parties

  Inside directory `build`, run:

  ```shell
  cd build && cmake -DWITH_XLA=ON -DTHIRD_PARTY=ON ..
  make -j$(nproc)
  ```

  If the downloading error occurred, you should go back to the previous step to reinstall the cmake, then clean the file CMakeCache.txt and build the third-parties once again.

- Build OneFlow

  Inside directory `build`, run:
  ```shell
  cmake .. \
  -DWITH_XLA=ON \
  -DTHIRD_PARTY=OFF \
  -DPython_NumPy_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
  -DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
  -DPYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])")
  
  make -j$(nproc)
  ```

### Build with TensorRT

- Build third-parties

  1. Download TensorRT(>=6.0) .tgz and unzip the package.
  
  2. Inside directory `build`, run:
  
  ```shell
  cmake -DWITH_TENSORRT=ON -DTENSORRT_ROOT=your_tensorrt_path -DTHIRD_PARTY=ON ..
  make -j$(nproc)
  ```
- Build OneFlow

  Inside directory `build`, run:
  ```shell
  cmake .. \
  -DWITH_TENSORRT=ON \
  -DTENSORRT_ROOT=your_tensorrt_path \
  -DTHIRD_PARTY=OFF \
  -DPython_NumPy_INCLUDE_DIRS=$(python3 -c "import numpy; print(numpy.get_include())") \
  -DPYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])") \
  -DPYTHON_LIBRARY=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['stdlib'])")

  make -j$(nproc)
  ```

### Documents

 - XRT documents

   You can check this [doc](./oneflow/xrt/README.md) to obtain more details about how to use XLA and TensorRT with OneFlow.
