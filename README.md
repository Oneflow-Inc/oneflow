# OneFlow

OneFlow is a deep learning framework designed to be **user-friendly, scalable and efficient**. With OneFlow, it is easy to:

- program a model with **PyTorch-like API**
- scale a model to n-dimensional-parallel/distributed execution with the **Global View API**
- accelerate/deploy a model with the **Static Graph Compiler**.

[![Simple CI](https://github.com/Oneflow-Inc/oneflow/actions/workflows/simple.yml/badge.svg)](https://github.com/Oneflow-Inc/oneflow/actions/workflows/simple.yml)
[![Nightly Docker Image](https://github.com/Oneflow-Inc/docker-images/actions/workflows/oneflow-nightly.yml/badge.svg)](https://github.com/Oneflow-Inc/docker-images/actions/workflows/oneflow-nightly.yml)
[![Nightly Release](https://github.com/Oneflow-Inc/oneflow/actions/workflows/release.yml/badge.svg)](https://github.com/Oneflow-Inc/oneflow/actions/workflows/release.yml)
[![Documentation](https://readthedocs.org/projects/oneflow/badge/?version=master)](https://oneflow.readthedocs.io/en/master/?badge=master)

## Latest News

- Version 0.9.0 is out!
  - [Full changelog](https://github.com/Oneflow-Inc/oneflow/releases/tag/v0.9.0)

## Publication

- [OneFlow: Redesign the Distributed Deep Learning Framework from Scratch](https://arxiv.org/abs/2110.15032)
- Bibtex Citation

  ```
  @misc{yuan2021oneflow,
        title={OneFlow: Redesign the Distributed Deep Learning Framework from Scratch},
        author={Jinhui Yuan and Xinqi Li and Cheng Cheng and Juncheng Liu and Ran Guo and Shenghang Cai and Chi Yao and Fei Yang and Xiaodong Yi and Chuan Wu and Haoran Zhang and Jie Zhao},
        year={2021},
        eprint={2110.15032},
        archivePrefix={arXiv},
        primaryClass={cs.DC}
  }
  ```

## Install OneFlow

### System Requirements

- Linux. As for now, there is no pre-built release for macOS, Windows.
- Python 3.7, 3.8, 3.9, 3.10
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
  python3 -m pip install oneflow
  ```

- To install nightly release of OneFlow with CUDA support:

  ```bash
  python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu117
  ```

- To install other available builds for different variants:

  - Stable
    ```bash
    python3 -m pip install --find-links https://release.oneflow.info oneflow==0.9.0+cu117
    ```
  - Nightly
    ```
    python3 -m pip install --pre oneflow -f https://staging.oneflow.info/branch/master/[PLATFORM]
    ```
  - All available `[PLATFORM]`:
    | Platform |CUDA Driver Version| Supported GPUs |
    |---|---|---|
    | cu117 | >= 450.80.02 | GTX 10xx, RTX 20xx, A100, RTX 30xx |
    | cu102 | >= 440.33 | GTX 10xx, RTX 20xx |
    | cpu | N/A | N/A |

- If you are in China, you could run this to have pip download packages from domestic mirror of pypi:
  ```
  python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```
  For more information on this, please refer to [pypi 镜像使用帮助](https://mirror.tuna.tsinghua.edu.cn/help/pypi/)

### Use docker image

```
docker pull oneflowinc/oneflow:nightly-cuda11.7
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

  - Pull the docker image:

    ```bash
    docker pull oneflowinc/manylinux2014_x86_64_cuda11.2
    ```

  - Follow the instructions in the bare metal build guide below.

- #### Option 3: Build on bare metal

  - Install dependencies (not required if you are using docker):
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

- [OneFlow-XRT](https://github.com/Oneflow-Inc/oneflow-xrt): An extension for OneFlow to target third-party compiler, such as XLA, TensorRT and OpenVINO etc.

## Getting Started

- Please refer to [QUICKSTART](https://docs.oneflow.org/en/master/basics/01_quickstart.html)
- 中文版请参见 [快速上手](https://docs.oneflow.org/master/basics/01_quickstart.html)

## Documentation

- [API Reference](https://oneflow.readthedocs.io/en/master/)
- [Usage & Design Docs](http://docs.oneflow.org/)
- [System Design](https://docs.oneflow.org/en/v0.4.0/basics_topics/essentials_of_oneflow.html)

## Model Zoo and Benchmark

- [Libai(Toolbox for Parallel Training Large-Scale Transformer Models)](https://github.com/Oneflow-Inc/libai)
  - [BERT-large](https://libai.readthedocs.io/en/latest/tutorials/get_started/quick_run.html)
  - [GPT](https://libai.readthedocs.io/en/latest/modules/libai.models.html#id5)
  - [T5](https://libai.readthedocs.io/en/latest/modules/libai.models.html#id4)
  - [VisionTransformer](https://libai.readthedocs.io/en/latest/modules/libai.models.html#id1)
  - [SwinTransformer](https://libai.readthedocs.io/en/latest/modules/libai.models.html#id2)
- [FlowVision(Toolbox for Computer Vision Datasets, SOTA Models and Utils)](https://github.com/Oneflow-Inc/vision)
- [OneFlow-Models(Examples of How to Implement Models in Various Fields with OneFlow)](https://github.com/Oneflow-Inc/models)
  - [ResNet-50](https://github.com/Oneflow-Inc/models/tree/main/Vision/classification/image/resnet50)
  - [Wide&Deep](https://github.com/Oneflow-Inc/models/tree/main/RecommenderSystems/wide_and_deep)
- [OneFlow-Benchmark(Outdated)](https://github.com/Oneflow-Inc/OneFlow-Benchmark)

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
