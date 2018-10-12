# Oneflow

### 1.1 Linux 

```
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
mkdir build && cd build
cmake -DBUILD_THIRD_PARTY=ON .. && make
cmake -DBUILD_THIRD_PARTY=OFF .. && make
```
### Build

Building OneFlow from source requires a `BLAS libary` installed. On CentOS, if you have `Intel MKL` installed, please update the environment variable. 

```
    export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
```

Or you can install OpenBLAS through: 

```
    sudo yum install epel-release & sudo yum install openblas-devel
```

#### download third party

```
    cmake -DTHIRD_PARTY=ON .. &&  make
    cmake -DTHIRD_PARTY=OFF .. && make
```

#### build third party

```
    cmake -DTHIRD_PARTY=ON -DDOWNLOAD_THIRD_PARTY=OFF .. &&  make
    cmake -DTHIRD_PARTY=OFF -DDOWNLOAD_THIRD_PARTY=OFF .. && make
```

