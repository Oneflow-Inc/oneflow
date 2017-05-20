del build
mkdir build
cd ./build
cmake .. -A x64 -DBUILD_THIRD_PARTY=ON
msbuild /p:Configuration=Debug prepare_oneflow_third_party.vcxproj
cd ..

