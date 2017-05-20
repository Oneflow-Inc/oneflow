rm -rf build
mkdir build
cd ./build
cmake .. -DBUILD_THIRD_PARTY=ON
make -j
cd ..

