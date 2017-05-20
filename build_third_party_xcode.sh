rm -rf build
mkdir build
cd ./build
cmake .. -G Xcode -DCMAKE_BUILD_TYPE=Debug -DBUILD_THIRD_PARTY=ON
make -j
rm CMakeCache.txt
cd ..

