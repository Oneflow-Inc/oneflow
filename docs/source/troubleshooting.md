# Troubleshooting

- Failed to compile `.cu` files
    1. Please refer to [CUDA System Requirements](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements) . Make sure your linux distribution and libraries shipped with it meet the requirements.
    2. If you are using tools like conda, please make sure libraries you install doesn't shade the proper installation comes with linux distribution or package management like apt-get.
    3. Please build OneFlow with a newer version of CMake. You could download version 3.14 from here: [https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.tar.gz](https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.tar.gz)

- How do I know what compilers and flags are used to compile OneFlow?
    - run `make clean && make VERBOSE=1` to get exact compile commands with compiler path and flags

- How to compile OneFlow with RDMA support?
    - add cmake flag `-DBUILD_RDMA` to compile OneFlow

- SWIG not found
    - Usually you could install it with a package manager like apt-get. You can also build it from source. Refer to [SWIG official release](http://www.swig.org/download.html)

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

- failed to compile because C++ 17 is enabled
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
