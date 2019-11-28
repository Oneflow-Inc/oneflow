# Oneflow

### 1.1 Linux 

### Build

Building OneFlow from source requires a `BLAS libary` installed. On CentOS, if you have `Intel MKL` installed, please update the environment variable. 

```
    export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
```

Or you can install OpenBLAS and other tools through:

```
    sudo yum -y install epel-release && sudo yum -y install git gcc-c++ cmake3 openblas-devel kernel-devel-$(uname -r) nasm
```

#### clone source code

> note: with `--recursive` flag to clone third_party submodules

```
    git clone https://github.com/Oneflow-Inc/oneflow --recursive
```

or you can just clone source code and submodules step by step

```
    git clone https://github.com/Oneflow-Inc/oneflow
    git submodule update --init --recursive
```

#### build third party from source

```
  cmake -DTHIRD_PARTY=ON .. && make -j
```

#### build oneflow

```
    cmake -DTHIRD_PARTY=OFF .. && make -j
```

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

  Run the following command to build third-parties.

  ```shell
  cd build && cmake -DWITH_XLA=ON -DTHIRD_PARTY=ON ..
  make -j$(nproc)
  ```

  If the downloading error occurred, you should go back to the previous step to reinstall the cmake, then clean the file CMakeCache.txt and build the third-parties once again.

- Build OneFlow

  ```shell
  cmake .. \
  -DWITH_XLA=ON \
  -DPYTHON_LIBRARY=your_python_lib_path \
  -DPYTHON_INCLUDE_DIR=your_python_include_dir \
  -DPython_NumPy_INCLUDE_DIRS=your_numpy_include_dir
  
  make -j$(nproc)
  ```

- XLA documents

  You can check this [doc](./oneflow/xrt/README.md) to obtain more details about how to use XLA.

### Build with TensorRT

- Build third-parties

  Run the following command to build third-parties.

  ```shell
  cd build && cmake -DWITH_TENSORRT=ON -DTHIRD_PARTY=ON ..
  make -j$(nproc)
  ```
- Build OneFlow

  ```shell
  cmake .. \
  -DWITH_TENSORRT=ON \
  -DPYTHON_LIBRARY=your_python_lib_path \
  -DPYTHON_INCLUDE_DIR=your_python_include_dir \
  -DPython_NumPy_INCLUDE_DIRS=your_numpy_include_dir

  make -j$(nproc)
  ```
