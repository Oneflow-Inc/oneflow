# Oneflow

### 1.1 Linux 

```
export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
mkdir build && cd build
cmake -DBUILD_THIRD_PARTY=ON .. && make
cmake -DBUILD_THIRD_PARTY=OFF .. && make
```

### docker usage

#### build

1. `base image` & `git repo` pull

    ```bash
        docker pull nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

        git clone Oneflow-Inc/oneflow
    ```

2. build oneflow

    ```bash
        docker run -it -v `./path/to/oneflow`:/opt --rm nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
    ```

#### known issues

during build process:

* FindCMake.cmake looks for /usr/local/cuda. Just create a symbolic link of that name to your actual CUDA installation directory.
