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

#### create `build` directory:

```
    cd oneflow && mkdir build && cd build
```

#### build third party from source

```
    cmake -DTHIRD_PARTY=ON -DPRECOMPILED_THIRD_PARTY=OFF .. &&  make -j
```

If you do not want to compile the third part code from source, you can also download the pre-compiled third party dependencies from oneflow.org

#### download pre-compiled third party

```
    cmake -DTHIRD_PARTY=ON .. &&  make -j
```

#### build oneflow

```
    cmake -DTHIRD_PARTY=OFF .. && make -j
```
