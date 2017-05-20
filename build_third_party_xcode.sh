rm -rf build_third_party
mkdir build_third_party
cd ./build_third_party
cmake ../cmake/third_party -G Xcode -DCMAKE_BUILD_TYPE=Debug
make -j
cd ..

