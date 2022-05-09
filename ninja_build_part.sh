cd build && cmake .. \
        -DONEFLOW=OFF   \
        -DTHIRD_PARTY=ON   \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DBUILD_TESTING=OFF  \
        -DBUILD_PROFILER=OFF   \
        -DTREAT_WARNINGS_AS_ERRORS=OFF   \
        -DCUDNN_ROOT_DIR=/usr/local/cudnn  \
        -C  ../cmake/caches/cn/fast/cuda-75.cmake        \
        -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld"     \
        -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld"  \
        -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld"   && time ninja -j16
# -DCMAKE_C_COMPILER="clang"   \
# -DCMAKE_CXX_COMPILER="clang++"   \
# -DCMAKE_CUDA_HOST_COMPILER="clang++"   \



# [209/209] Running utility command for prepare_oneflow_third_party

# real    16m28.611s
# user    90m10.375s
# sys     4m3.496s
