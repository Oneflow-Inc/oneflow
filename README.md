# OneFlow

OneFlow is a deep learning framework designed to be **user-friendly, scalable and efficient**. With OneFlow, it is easy to:

- program a model with [**PyTorch-like API**](https://oneflow.readthedocs.io/en/master/)
- scale a model to n-dimensional-parallel execution with the [**Global Tensor**](https://docs.oneflow.org/en/master/cookies/global_tensor.html)
- accelerate/deploy a model with the [**Graph Compiler**](https://oneflow.readthedocs.io/en/master/graph.html).

[![Simple CI](https://github.com/Oneflow-Inc/oneflow/actions/workflows/simple.yml/badge.svg)](https://github.com/Oneflow-Inc/oneflow/actions/workflows/simple.yml)
[![Nightly Docker Image](https://github.com/Oneflow-Inc/docker-images/actions/workflows/oneflow-nightly.yml/badge.svg)](https://github.com/Oneflow-Inc/docker-images/actions/workflows/oneflow-nightly.yml)
[![Nightly Release](https://github.com/Oneflow-Inc/oneflow/actions/workflows/release.yml/badge.svg)](https://github.com/Oneflow-Inc/oneflow/actions/workflows/release.yml)
[![Documentation](https://readthedocs.org/projects/oneflow/badge/?version=master)](https://oneflow.readthedocs.io/en/master/?badge=master)

## Latest News

- Version 1.0.0 is out!
  - [Full changelog](https://github.com/Oneflow-Inc/oneflow/releases/tag/v1.0.0)

## Publication

- [OneFlow: Redesign the Distributed Deep Learning Framework from Scratch](https://arxiv.org/abs/2110.15032)

## System Requirements

### General
- Linux
- Python 3.7, 3.8, 3.9, 3.10, 3.11

### CUDA
- CUDA arch 60 or above
- CUDA Toolkit version 10.0 or above
- Nvidia driver version 440.33 or above

  OneFlow will work on a minimum supported driver, and any driver beyond. For more information, please refer to [CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).

## Install

### Preinstall docker image

```
docker pull oneflowinc/oneflow:nightly-cuda11.8
```

### Pip Install

- (**Highly recommended**) Upgrade pip

  ```
  python3 -m pip install --upgrade pip #--user
  ```

- To install latest stable release of OneFlow with CUDA support:

  ```bash
  python3 -m pip install oneflow
  ```

- To install nightly release of OneFlow with CPU-only support:

  ```bash
  python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cpu
  ```

- To install nightly release of OneFlow with CUDA support:

  ```bash
  python3 -m pip install --pre oneflow -f https://oneflow-staging.oss-cn-beijing.aliyuncs.com/branch/master/cu118
  ```

  If you are in China, you could run this to have pip download packages from domestic mirror of pypi:
  ```
  python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  ```
  For more information on this, please refer to [pypi 镜像使用帮助](https://mirror.tuna.tsinghua.edu.cn/help/pypi/)

### Install from Source

<details>
<summary>Clone Source Code</summary>

- #### Option 1: Clone source code from GitHub

  ```bash
  git clone https://github.com/Oneflow-Inc/oneflow.git
  ```

- #### Option 2: Download from Aliyun(Only available in China)

  ```bash
  curl https://oneflow-public.oss-cn-beijing.aliyuncs.com/oneflow-src.zip -o oneflow-src.zip
  unzip oneflow-src.zip
  ```

  </details>

<details>
<summary>Build OneFlow</summary>

- Install dependencies
  ```
  apt install -y libopenblas-dev nasm g++ gcc python3-pip cmake autoconf libtool
  ```
  These dependencies are preinstalled in offical conda environment and docker image, you can use the offical conda environment [here](https://github.com/Oneflow-Inc/conda-env) or use the docker image by:
  ```bash
  docker pull oneflowinc/manylinux2014_x86_64_cuda11.2
  ```
- In the root directory of OneFlow source code, run:

  ```
  mkdir build
  cd build
  ```

- Config the project, inside `build` directory:

  - If you are in China

    config for CPU-only like this:

    ```
    cmake .. -C ../cmake/caches/cn/cpu.cmake
    ```

    config for CUDA like this:

    ```
    cmake .. -C ../cmake/caches/cn/cuda.cmake -DCMAKE_CUDA_ARCHITECTURES=80 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DCUDNN_ROOT_DIR=/usr/local/cudnn
    ```

  - If you are not in China

    config for CPU-only like this:

    ```
    cmake .. -C ../cmake/caches/international/cpu.cmake
    ```

    config for CUDA like this:

    ```
    cmake .. -C ../cmake/caches/international/cuda.cmake -DCMAKE_CUDA_ARCHITECTURES=80 -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DCUDNN_ROOT_DIR=/usr/local/cudnn
    ```
    Here the DCMAKE\_CUDA\_ARCHITECTURES macro is used to specify the CUDA architecture, and the DCUDA\_TOOLKIT\_ROOT\_DIR and DCUDNN\_ROOT\_DIR macros are used to specify the root path of the CUDA Toolkit and CUDNN.

  - For AMD GPUs with ROCm, use the CUDA cache (from either directory) and enable HIP, compiling with the ROCm clang:

    ```
    cmake .. -C ../cmake/caches/international/cuda.cmake -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx942 -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ -DWITH_CUTLASS=OFF -DWITH_MLIR=OFF -DBUILD_RDMA=OFF -DBUILD_CPP_API=OFF -DBUILD_HWLOC=OFF -DTREAT_WARNINGS_AS_ERRORS=OFF -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
    ```
    Here the DCMAKE\_HIP\_ARCHITECTURES macro specifies the GPU architecture; when it is omitted, the GPUs of the build machine are detected. One build covers either wavefront-64 GPUs (gfx9, e.g. MI200/MI300) or wavefront-32 GPUs (gfx10/gfx11), not both. `build_rocm.sh` in the root directory configures and builds with these options (Ninja, Release). With ROCm, the cuDNN, cuBLASLt, cuSOLVER and nvJPEG/NPP code paths run on MIOpen, hipBLASLt, hipSOLVER and rocJPEG/RPP; kernels that need CUTLASS are not built, as in a CUDA build with -DWITH\_CUTLASS=OFF.

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
- [OneFlow-Models(Outdated)](https://github.com/Oneflow-Inc/models)
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
