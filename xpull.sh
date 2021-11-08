git pull test start-serial 

cd build 

cmake .. -C ../cmake/caches/cn/cuda.cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_RDMA=ON  -DCUDNN_ROOT_DIR=/usr/local/cudnn

make -j$(nproc)