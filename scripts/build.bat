@echo off
if "%~1" == "third_party" (
  del build
  mkdir build
  cd ./build
  cmake .. -A x64 -DBUILD_THIRD_PARTY=ON -DCMAKE_BUILD_TYPE=Debug -DCUDNN_INCLUDE_DIR="D:\users\jiyuan\cudnn5.1\include" -DCUDNN_LIBRARY="D:\users\jiyuan\cudnn5.1\lib\x64\cudnn.lib"
  msbuild /p:Configuration=Debug prepare_oneflow_third_party.vcxproj
  cd ..
  ) else (
  if "%~1" == "" (
  	rem mkdir build
    cd ./build
    cmake .. -A x64 -DBUILD_THIRD_PARTY=OFF -DCMAKE_BUILD_TYPE=Debug -DCUDNN_INCLUDE_DIR="D:\users\jiyuan\cudnn5.1\include" -DCUDNN_LIBRARY="D:\users\jiyuan\cudnn5.1\lib\x64\cudnn.lib"
    msbuild /p:Configuration=Debug ALL_BUILD.vcxproj
    cd ..
  	)
  )


