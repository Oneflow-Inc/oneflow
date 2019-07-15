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
