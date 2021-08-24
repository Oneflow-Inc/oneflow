cd build 
cmake .. -DTHIRD_PARTY_MIRROR=aliyun -DCMAKE_BUILD_TYPE=RelWithDebInfo   -DBUILD_RDMA=ON  -DTREAT_WARNINGS_AS_ERRORS=OFF

make -j$(nproc)