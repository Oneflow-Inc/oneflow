@echo off
if "%~1" == "third_party" (
  del build
  mkdir build
  cd ./build
  cmake .. -A x64 -DBUILD_THIRD_PARTY=ON
  msbuild /p:Configuration=Debug prepare_oneflow_third_party.vcxproj
  cd ..
  ) else (
  if "%~1" == "" (
  	rem mkdir build
    cd ./build
    cmake .. -A x64
    msbuild /p:Configuration=Debug ALL_BUILD.vcxproj
    cd ..
  	)
  )


