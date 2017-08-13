if [ "$1" = "third_party" ]
then
  rm -rf build
  mkdir build
  cd ./build
  cmake .. -G Xcode -DCMAKE_BUILD_TYPE=Debug -DBUILD_THIRD_PARTY=ON
  make -j
  cd ..
elif [ "$1" = "" ]
then
  cd ./build
  cmake .. -G Xcode -DCMAKE_BUILD_TYPE=Debug
  make -j
  cd ..
fi