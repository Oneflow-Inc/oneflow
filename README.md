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

  Download and Install bazel from [here](https://docs.bazel.build/versions/master/install-os-x.html) , and version 0.24.1 is recommended. You can confirm bazel is installed successfully by running the following command:

  ```shell
  bazel version
  ```

- Update cmake

  This is only needed if the installed cmake does not support https protocol to download .tgz from URL. You can go back and reinstall cmake if you had encountered an downloading error while building the third-parties.

  Download cmake from [here](https://cmake.org/download/) , configure and install it by the following command:

  ```shell
  # Install curl develop toolkit
  sudo yum install libcurl-devel
 
  # install cmake
  cd cmake && ./bootstrap --system-curl --prefix=$your_path && make install
  ```

- Build third-parties

  Run the following command to build third-parties.

  ```shell
  nproc=`cat /proc/cpuinfo | grep "processor" | wc -l`
  cd build && cmake -DTHIRD_PARTY=ON ..
  make -j$nproc
  ```

  If the downloading error occurred, you should go back to the previous step to reinstall the cmake, then clean the file CMakeCache.txt and build the third-parties again.

- Build oneflow

  ```shell
  cmake .. \
  -DWITH_XLA=ON \
  -DPYTHON_LIBRARY=your_python_lib_path \
  -DPYTHON_INCLUDE_DIR=your_python_include_dir \
  -DPython_NumPy_INCLUDE_DIRS=your_numpy_include_dir
  
  make -j$nproc
  ```

- XLA doc

  You can check this [doc]() to obtain more details abot how to use XLA.

