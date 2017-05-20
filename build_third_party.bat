del build_third_party
mkdir build_third_party
cd ./build_third_party
cmake ../cmake/third_party -A x64
msbuild /p:Configuration=Debug ALL_BUILD.vcxproj
cd ..

