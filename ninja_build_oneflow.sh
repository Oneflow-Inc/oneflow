cd build && cmake ..                       \
        -DONEFLOW=ON                       \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo  \
        -DTHIRD_PARTY=OFF                  \
        -DBUILD_TESTING=OFF                \
        -DBUILD_PROFILER=OFF               \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1  \
        -DTREAT_WARNINGS_AS_ERRORS=OFF     \
        -C  ../cmake/caches/cn/fast/cuda-75.cmake          \
        -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld"       \
        -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld"    \
        -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld" && time ninja -j32
# -DCMAKE_C_COMPILER="clang"   \
# -DCMAKE_CXX_COMPILER="clang++"   \
# -DCMAKE_CUDA_HOST_COMPILER="clang++"   \


# [1753/1753] Running utility command for of_include_copy


# all(oneflow+third part) ninja compile cost
# real    50m48.089s
# user    275m27.361s
# sys     15m11.990s

# oneflow ninja compile cost
# [1544/1544] Running utility command for of_include_copy
# real    27m22.249s
# user    143m0.141s
# sys     8m49.611s

# oneflow(增量编译) ninja compile cost
# real    0m10.897s
# user    0m17.332s
# sys     0m2.834s