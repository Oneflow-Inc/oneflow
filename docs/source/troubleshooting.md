# Troubleshooting

- `CUDNN_STATUS_NOT_INITIALIZED`
    - You might see error message like these:
        ```
        I0729 22:37:45.483937439   56788 ev_epoll_linux.c:82]        Use of signals is disabled. Epoll enginll not be used
        E0729 22:37:45.515343 56788 version.cpp:82] Failed to get cuda runtime version: CUDA driver version nsufficient for CUDA runtime version
        F0729 22:38:31.209002 56788 improver.cpp:535] Check failed: mem_size > 0 (-524288000 vs. 0)
        ```
        ```
        F0723 19:05:56.194067 40970 cuda_util.cpp:82] Check failed: error == CUDNN_STATUS_SUCCESS (1 vs. 0) CUDNN_STATUS_NOT_INITIALIZED
        ```
    - Please upgrade to Nvidia Linux x86_64 driver. Version >= 440.33 is recommended.
    - For more information, please refer to [CUDA compatibility documentation](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).

- Failed to compile `.cu` files
    - Please refer to [CUDA System Requirements](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) . Make sure your linux distribution and libraries shipped with it meet the requirements.
    - If you are using tools like conda, please make sure libraries you install doesn't shade the proper installation comes with linux distribution or package management like apt-get.
    - Please build OneFlow with a newer version of CMake. You could download version 3.14 from here: [https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.tar.gz](https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.tar.gz)

- How do I know what compilers and flags are used to compile OneFlow?
    - run `make clean && make VERBOSE=1` to get exact compile commands with compiler path and flags

- How to compile OneFlow with RDMA support?
    - add cmake flag `-DBUILD_RDMA` to compile OneFlow

- Which version of g++ CMake is using to build OneFlow?
    - You should find a line like this in CMake output:

        ```bash
        -- CMAKE_CXX_COMPILER_VERSION: [YOUR G++ VERSION NUMBER]
        ```

- Failed to compile NCCL
    - Try use less threads when compiling OneFlow third party. For instance, use

        ```bash
        cmake -DTHIRD_PARTY=ON .. && make
        ```

        instead of

        ```bash
        cmake -DTHIRD_PARTY=ON .. && make -j$(nproc) `
        ```

- `"CUDA_VERSION" "VERSION_GREATER_EQUAL" "10.0"`
    - Please use a newer version of CMake
    - Make sure cmake is correctly included in `PATH`

- CUBLAS not found
    - Usually it happens when using CUDA 10.1 or newer
    - You should see error massage by CMake like this:

        ```
        cuda lib not found: /usr/local/miniconda3/envs/dl/lib/libcublas_static.a or
        /usr/local/cuda/lib64/libcublas_static.a
        ```

    - Make sure `libcublas_static.a` is in one of the two directories.

- When running OneFlow in gdb, there is no debug information for code location.
    - add cmake flag `-DCMAKE_BUILD_TYPE=RELWITHDEBINFO` or `-DCMAKE_BUILD_TYPE=DEBUG` and recompile

- `libof_ccobj.a: File truncated`
    - You might see error message like this:

        ```
        /usr/bin/ar: libof_ccobj.a: File truncated
        make[2]: *** [libof_ccobj.a] Error 1
        make[2]: *** Deleting file `libof_ccobj.a'
        make[1]: *** [CMakeFiles/of_ccobj.dir/all] Error 2
        make: *** [all] Error 2
        ```

    - You should upgrade your GNU Binutils. Version 2.33.1 is recommended. If you are using conda, you could install it by running `conda install -c conda-forge binutils`

- Failed to compile because C++ 17 is enabled
    - In some cases, environment variable `CXXFLAGS` is not empty and contains `--std c++17`.
    - Check if it is empty by running `echo $CXXFLAGS` and clear it with `unset CXXFLAGS`.

- cmake outputs error `No CMAKE_ASM_NASM_COMPILER could be found.`
    - Install `nasm`. For instance, run `sudo yum install nasm` if you are on centos.

- `No module named 'google.protobuf'`
    - You might see error message like this:
        ```
        Scanning dependencies of target generate_api
        ...
            from google.protobuf import descriptor as _descriptor
        ModuleNotFoundError: No module named 'google.protobuf'
        CMakeFiles/generate_api.dir/build.make:57: recipe for target 'CMakeFiles/generate_api' failed
        make[2]: *** [CMakeFiles/generate_api] Error 1
        ```
    - Install development dependencies by running:
        ```
        pip3 install -r dev-requirements.txt
        ```

- Get gdb warning `ptrace: Operation not permitted.` and gdb command `bt` prints no backtrace
    - You might get this warning when debugging OneFlow with gdb inside a docker container. Try add these flags when launching your container:
        ```
        docker run --cap-add=SYS_PTRACE --security-opt seccomp=unconfined
        ```
    - Please refer to https://stackoverflow.com/questions/19215177/how-to-solve-ptrace-operation-not-permitted-when-trying-to-attach-gdb-to-a-pro

- It takes too long to download python packages when running `make`
    - If you are in China, you could run this to have pip download packages from domestic mirror of pypi:
        ```
        python3 -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
        ```
    - For more information on this, please refer to [pypi 镜像使用帮助](https://mirror.tuna.tsinghua.edu.cn/help/pypi/)
