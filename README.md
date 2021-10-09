# OneFlow

**OneFlow is a performance-centered and open-source deep learning framework.**

[![Simple CI](https://github.com/Oneflow-Inc/oneflow/actions/workflows/simple.yml/badge.svg)](https://github.com/Oneflow-Inc/oneflow/actions/workflows/simple.yml)
[![Documentation Status](https://readthedocs.org/projects/oneflow/badge/?version=master)](https://oneflow.readthedocs.io/en/master/?badge=master)

## Latest News

- Version 0.5.0 is out!
  - First class support for eager execution. The deprecated APIs are moved to `oneflow.compatible.single_client`
  - Drop-in replacement of `import torch` for existing Pytorch projects. You could test it by inter-changing `import oneflow as torch` and `import torch as flow`.
  - [Full changelog](https://github.com/Oneflow-Inc/oneflow/releases/tag/v0.5.0)

## Install OneFlow

### System Requirements

- Python 3.6, 3.7, 3.8, 3.9
- (**Highly recommended**) Upgrade pip

  ```
  python3 -m pip install --upgrade pip #--user
  ```

- CUDA Toolkit Linux x86_64 Driver

  - CUDA runtime is statically linked into OneFlow. OneFlow will work on a minimum supported driver, and any driver beyond. For more information, please refer to [CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).

  - Please upgrade your Nvidia driver to version 440.33 or above and install OneFlow for CUDA 10.2 if possible.

### Install with Pip Package

- To install latest stable release of OneFlow with CUDA support:

  ```bash
  python3 -m pip install -f https://release.oneflow.info oneflow==0.5.0+cu102
  ```

- To install nightly release of OneFlow with CUDA support:

  ```bash
  python3 -m pip install oneflow -f https://staging.oneflow.info/branch/master/cu102
  ```

- To install other available builds for different variants:

  - Stable
    ```bash
    python3 -m pip install --find-links https://release.oneflow.info oneflow==0.5.0+[PLATFORM]
    ```
  - Nightly
    ```
    python3 -m pip install oneflow -f https://staging.oneflow.info/branch/master/[PLATFORM]
    ```
  - All available `[PLATFORM]`:
    | Platform |CUDA Driver Version| Supported GPUs |
    |---|---|---|
    | cu112 | >= 450.80.02 | GTX 10xx, RTX 20xx, A100, RTX 30xx |
    | cu111 | >= 450.80.02 | GTX 10xx, RTX 20xx, A100, RTX 30xx |
    | cu110, cu110_xla | >= 450.36.06 | GTX 10xx, RTX 20xx, A100|
    | cu102, cu102_xla | >= 440.33 | GTX 10xx, RTX 20xx |
    | cu101, cu101_xla | >= 418.39 | GTX 10xx, RTX 20xx |
    | cu100, cu100_xla | >= 410.48 | GTX 10xx, RTX 20xx |
    | cpu | N/A | N/A |

- If you are in China, you could run this to have pip download packages from domestic mirror of pypi:
  ```
  python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```
  For more information on this, please refer to [pypi 镜像使用帮助](https://mirror.tuna.tsinghua.edu.cn/help/pypi/)

### Use docker image

```
docker pull oneflowinc/oneflow:nightly-cuda10.2
docker pull oneflowinc/oneflow:nightly-cuda11.1
```

### Build from Source

<details>
<summary>Clone Source Code</summary>

- #### Option 1: Clone source code from GitHub

  ```bash
  git clone https://github.com/Oneflow-Inc/oneflow --depth=1
  ```

- #### Option 2: Download from Aliyun

  If you are in China, please download OneFlow source code from: https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-src.zip

  ```bash
  curl https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-src.zip -o oneflow-src.zip
  unzip oneflow-src.zip
  ```

  </details>

<details>
<summary>Build OneFlow</summary>

- #### Option 1: Build with Conda (recommended)

  Please refer to [this repo](https://github.com/Oneflow-Inc/conda-env)

- #### Option 2: Build in docker container (recommended)

  - Pull a docker image:

    ```
    docker pull oneflowinc/oneflow-manylinux2014-cuda10.2:0.1
    ```

    All images available : https://hub.docker.com/u/oneflowinc

  - In the root directory of OneFlow source code, run:

    ```
    python3 docker/package/manylinux/build_wheel.py --python_version=3.6
    ```

    This should produce `.whl` files in the directory `wheelhouse`

  - If you are in China, you might need to add these flags:

    ```
    --use_tuna --use_system_proxy --use_aliyun_mirror
    ```

  - You can choose CUDA/Python versions of wheel by adding:

    ```
    --cuda_version=10.1 --python_version=3.6,3.7
    ```

  - For more useful flags, plese run the script with flag `--help` or refer to the source code of the script.

- #### Option 3: Build on bare metal

  - Install dependencies
    - on Ubuntu 20.04, run:
      ```
      sudo apt install -y libopenblas-dev nasm g++ gcc python3-pip cmake autoconf libtool
      ```
    - on macOS, run:
      ```
      brew install nasm
      ```
  - In the root directory of OneFlow source code, run:

    ```
    mkdir build
    cd build
    ```

  - Config the project, inside `build` directory:

    - If you are in China

      run this to config for CUDA:

      ```
      cmake .. -C ../cmake/caches/cn/cuda.cmake
      ```

      run this to config for CPU-only:

      ```
      cmake .. -C ../cmake/caches/cn/cpu.cmake
      ```

    - If you are not in China

      run this to config for CUDA:

      ```
      cmake .. -C ../cmake/caches/international/cuda.cmake
      ```

      run this to config for CPU-only:

      ```
      cmake .. -C ../cmake/caches/international/cpu.cmake
      ```

  - Build the project, inside `build` directory, run:

    ```
    make -j$(nproc)
    ```

  - Add oneflow to your PYTHONPATH, inside `build` directory, run:

    ```
    source source.sh
    ```

    Please note that this change is not permanent.

  - Simple validation

        ```
        python3 -m oneflow --doctor
        ```

    </details>

### Troubleshooting

Please refer to [troubleshooting](docs/source/troubleshooting.md) for common issues you might encounter when compiling and running OneFlow.

### Advanced features

<details>
<summary>XRT</summary>

- You can check this [doc](oneflow/xrt/README.md) to obtain more details about how to use XLA and TensorRT with OneFlow.
</details>

## Getting Started

<details>
<summary>3 minutes to run MNIST.</summary>

- Clone the demo code from OneFlow documentation

  ```
  git clone https://github.com/Oneflow-Inc/oneflow-documentation.git
  cd oneflow-documentation/cn/docs/single_client/code/quick_start/
  ```

- Run it in Python

  ```
  python mlp_mnist.py
  ```

- Oneflow is running and you got the training loss
  ```
  2.7290366
  0.81281316
  0.50629824
  0.35949975
  0.35245502
  ...
  ```
- More info on this demo, please refer to [doc on quick start](https://docs.oneflow.org/master/single_client/quick_start/quickstart_in_3_min.html).
</details>

## Documentation

- [API Reference](https://oneflow.readthedocs.io/en/master/)
- [Usage & Design Docs](http://docs.oneflow.org/)
- [System Design](https://docs.oneflow.org/en/v0.4.0/basics_topics/essentials_of_oneflow.html)

## Model Zoo and Benchmark

- [OneFlow Models](https://github.com/Oneflow-Inc/models)
- [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark)
- [GPT](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/LanguageModeling/GPT)
- [ResNet-50](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/resnet50)
- [Wide&Deep](https://github.com/Oneflow-Inc/models/tree/main/RecommenderSystems/wide_and_deep)
- [BERT](https://github.com/Oneflow-Inc/models/tree/main/NLP/bert-oneflow)

## Communication

- [GitHub issues](https://github.com/Oneflow-Inc/oneflow/issues): any install, bug, feature issues.
- [www.oneflow.org](http://www.oneflow.org): brand related information.

- ### 中文

  - QQ 群: 331883
  - 微信号（加好友入交流群）: OneFlowXZS
  - [知乎](https://www.zhihu.com/org/oneflow-17)

- ### International
  - [Discord](https://discord.gg/4kpjGA5bZY)
  - [Twitter](https://twitter.com/OneFlowNews)
  - [LinkedIn](https://www.linkedin.com/company/oneflow-inc)
  - [Medium](https://oneflow2020.medium.com)

## The Team

OneFlow was originally developed by [OneFlow Inc](http://www.oneflow.org) and [Zhejiang Lab](http://www.zhejianglab.com/).

## License

[Apache License 2.0](LICENSE)
