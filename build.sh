# cpu
# cmake -B build -C cmake/caches/ci/cpu.cmake -DWITH_MLIR=OFF -DBUILD_PROFILER=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DBUILD_RDMA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo  && cd build && time ninja -j32

# npu
cmake -B build -C cmake/caches/cn/npu.cmake -DWITH_MLIR=OFF -DBUILD_PROFILER=OFF -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DBUILD_RDMA=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo  && cd build && make -j32