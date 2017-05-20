rm -rf build_third_party
mkdir build_third_party
cd ./build_third_party
cmake ../cmake/third_party
make -j
cd ..

