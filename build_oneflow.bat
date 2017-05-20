mkdir build
cd ./build
cmake .. -A x64
msbuild /p:Configuration=Debug ALL_BUILD.vcxproj
cd ..


