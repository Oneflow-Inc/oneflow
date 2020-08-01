**OneFlow is a performance-centered and open-source platform for machine learning.**

- [Install OneFlow](#install-oneflow)
  - [System Requirements](#system-requirements)
  - [Install with Pip Package](#install-with-pip-package)
  - [Build from Source](#build-from-source)
  - [Troubleshooting](#troubleshooting)
  - [Advanced features](#advanced-features)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
    - [Usage & Design Docs](#usage--design-docs)
    - [API Reference](#api-reference)
- [Model Zoo and Benchmark](#model-zoo-and-benchmark)
  - [CNNs(ResNet-50, VGG-16, Inception-V3, AlexNet)](#cnnsresnet-50-vgg-16-inception-v3-alexnet)
  - [Wide&Deep](#widedeep)
  - [BERT](#bert)
- [Communication](#communication)
- [Contributing](#contributing)
- [The Team](#the-team)
- [License](#license)

## Install OneFlow

  ### System Requirements

  - Python >= 3.5
  - Nvidia Linux x86_64 driver version >= 440.33

  ### Install with Pip Package

  - To install latest release of OneFlow with CUDA support:

    ```
    python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu102 --user
    ```

  - To install OneFlow with legacy CUDA support, run one of:
    ```
    python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu101 --user
    python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu100 --user
    python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu92 --user
    python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu91 --user
    python3 -m pip install --find-links https://oneflow-inc.github.io/nightly oneflow_cu90 --user
    ```

  - Support for latest stable version of CUDA will be prioritized. Please upgrade your Nvidia driver to version 440.33 or above and install `oneflow_cu102` if possible. For more information, please refer to [CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).

  - CPU-only OneFlow is not available for now.

  - Releases are built with G++/GCC 4.8.5, cuDNN 7 and MKL 2020.0-088.

### Build from Source

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
    cd oneflow
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

### Advanced features

- #### XRT

  You can check this [doc](oneflow/xrt/README.md) to obtain more details about how to use XLA and TensorRT with OneFlow.

## Getting Started
3 minutes to run MNIST.
1. Clone the demo code from OneFlow documentation
```
git clone https://github.com/Oneflow-Inc/oneflow-documentation.git
cd oneflow-documentation/cn/docs/code/quick_start/
```
2. Run it in Python
```
python mlp_mnist.py
```

3. Oneflow is running and you got the training loss
```
2.7290366
0.81281316
0.50629824
0.35949975
0.35245502
...
```
More info on this demo, please refer to [doc on quick start](http://docs.oneflow.org/quick_start/quickstart_in_3_min.html).

## Documentation
#### Usage & Design Docs
* [link](http://docs.oneflow.org/)
#### API Reference
* [link](https://oneflow-api.readthedocs.io/en/latest/)

## Model Zoo and Benchmark
* [link](https://github.com/Oneflow-Inc/OneFlow-Benchmark)
### CNNs(ResNet-50, VGG-16, Inception-V3, AlexNet)
* [CNNs](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/of_develop_py3/Classification/cnns)

### Wide&Deep
* [OneFlow-WDL](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/of_develop_py3/ClickThroughRate/WideDeepLearning)
### BERT
* [Bert](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/of_develop_py3/LanguageModeling/BERT)

## Communication
* Github issues : any install, bug, feature issues.
* [www.oneflow.org](http://www.oneflow.org) : brand related information.

## Contributing
*  [link](http://docs.oneflow.org/contribute/intro.html)

## The Team
OneFlow was originally developed by [OneFlow Inc](http://www.oneflow.org) and [Zhejiang Lab](http://www.zhejianglab.com/).

## License
[Apache License 2.0](LICENSE)
