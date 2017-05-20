del build_oneflow
mkdir build_oneflow
cd ./build_oneflow
cmake .. -A x64
msbuild /p:Configuration=Debug ALL_BUILD.vcxproj
cd ..


