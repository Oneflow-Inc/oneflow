# Oneflow

### Install OneFlow

  - To install latest release of OneFlow:

    ```
    pip install oneflow
    ```

  - To install nightly release of OneFlow:

    ```
    pip install --find-links https://oneflow-inc.github.io/nightly oneflow
    ```

### Build OneFlow from Source

1. #### System Requirements

    - Please use a newer version of CMake to build OneFlow. You could download cmake release from here: 
      https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.tar.gz

    - Building OneFlow from source requires a [BLAS](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) libary installed. 
    
      It is recommended to install [Intel MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) which provides APIs compatible with BLAS. Please refer to Intel's official guide on how to install MKL [here](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library/choose-download.html).

      On CentOS, if you have MKL installed, please update the environment variable.

      ```
      export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
      ```

      Or you can install OpenBLAS and other tools through:

      ```
      sudo yum -y install epel-release
      sudo yum -y install gcc-c++ openblas-devel kernel-devel-$(uname -r) nasm swig
      ```

2. #### Clone Source Code

    Clone source code and submodules (faster, recommended)

    ```
    git clone https://github.com/Oneflow-Inc/oneflow
    git submodule update --init --recursive
    ```

    Or you could also clone the repo with `--recursive` flag to clone third_party submodules together

    ```
    git clone https://github.com/Oneflow-Inc/oneflow --recursive
    ```

3. #### Install Python Dev Requirements

    To install development dependencies and linter tools, run:
    ```
    python3 -m pip install -r dev-requirements.txt --user
    ```

4. #### Build and Install OneFlow

    ```
    cd build
    cmake ..
    make -j$(nproc)
    make pip_install
    ```

### Troubleshooting

Please refer to [troubleshooting](docs/source/troubleshooting.md) for common issues you might encounter when compiling and running OneFlow.

### Advanced Features

- #### XRT

  You can check this [doc](oneflow/xrt/README.md) to obtain more details about how to use XLA and TensorRT with OneFlow.
