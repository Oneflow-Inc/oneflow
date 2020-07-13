# Oneflow
![Build and Test CI](https://github.com/Oneflow-Inc/oneflow/workflows/Build%20and%20Test%20CI/badge.svg?branch=develop)

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

    Building OneFlow from source requires a `BLAS libary` installed. On CentOS, if you have `Intel MKL` installed, please update the environment variable. 

    ```
    export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
    ```

    Or you can install OpenBLAS and other tools through:

    ```
    sudo yum -y install epel-release && sudo yum -y install git gcc-c++ cmake3 openblas-devel kernel-devel-$(uname -r) nasm
    ```

    It is recommended to install MKL. Please refer to Intel's official guide on how to install MKL [here](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library/choose-download.html)

    If installed CMake doesn't support https scheme, please install a release with support for it. You could download cmake release from here: https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.tar.gz

2. #### Clone Source Code

    Clone source code and submodules (faster, recommended)

    ```
    git clone https://github.com/Oneflow-Inc/oneflow
    git submodule update --init --recursive
    ```

    or you can also clone the repo with `--recursive` flag to clone third_party submodules together

    ```
    git clone https://github.com/Oneflow-Inc/oneflow --recursive
    ```

3. #### Install Python Dev Requirements

    To install development dependencies and linter tools, run:
    ```
    pip3 install -r dev-requirements.txt --user
    ```

4. #### Build OneFlow

    Enter Build Directory, run:

    ```
    cd build
    ```

    Inside directory `build`, run:
    ```
    cmake ..

    make -j$(nproc)
    ```

5. #### Install OneFlow

    In the root path of OneFlow repo, run:
    ```
    pip3 install -e . --user
    ```

    Alternatively, you can also install OneFlow by adding `build/python_scripts` to your `PYTHONPATH`:
    ```
    export PYTHONPATH=$HOME/oneflow/build/python_scripts:$PYTHONPATH
    ```

### Troubleshooting

Please refer to [troubleshooting](docs/source/troubleshooting.md) for common issues you might encounter when compiling oneflow from source.

### Advanced Features

- #### XRT

  You can check this [doc](./oneflow/xrt/README.md) to obtain more details about how to use XLA and TensorRT with OneFlow.
