#git push test start-epoll
#git remote set-utl test 
git push  upstream start-epoll
cd build 
cmake .. -C ../cmake/caches/cn/cuda.cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_RDMA=ON

make -j$(nproc)