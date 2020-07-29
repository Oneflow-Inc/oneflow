# Oneflow

### Install OneFlow

  #### System Requirements to Run OneFlow

  - Nvidia Linux x86_64 driver version >= 440.33

  #### Install Pip package

  - To install latest release of OneFlow:

    ```
    python3 -m pip install oneflow
    ```

  - To install nightly release of OneFlow:

    ```
    python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow
    ```

### Build OneFlow from Source

1. #### System Requirements to Build OneFlow

    - Please use a newer version of CMake to build OneFlow. You could download cmake release from [here](https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.tar.gz).

    - Please make sure you have G++ and GCC >= 4.8.5 installed. Clang is not supported for now.

    - To install dependencies, run:

      ```
      yum-config-manager --add-repo https://yum.repos.intel.com/setup/intelproducts.repo && \
      rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
      yum update -y && yum install -y epel-release && \
      yum install -y intel-mkl-64bit-2020.0-088 nasm swig rdma-core-devel
      ```

      On CentOS, if you have MKL installed, please update the environment variable:

      ```
      export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
      ```

      If you don't want to build OneFlow with MKL, you could install OpenBLAS:

      ```
      sudo yum -y install openblas-devel
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

3. #### Build and Install OneFlow

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
