git push test startepoll

cd build

cmake .. -C ../cmake/caches/cn/cuda.cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo  -DDBUILD_RDMA=ON


make -j$(nproc)