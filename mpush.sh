git push test start-callback 
cd build 
cmake .. -C ../cmake/caches/cn/cuda.cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_RDMA=ON

make -j$(nproc)