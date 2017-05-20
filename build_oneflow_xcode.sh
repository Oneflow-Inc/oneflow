rm -rf build_oneflow
mkdir build_oneflow
cd ./build_oneflow
cmake .. -G Xcode -DCMAKE_BUILD_TYPE=Debug
make -j
cd ..


