if [ "$1" = "third_party" ]
then
  rm -rf build
  mkdir build
  cd ./build
  cmake .. -DBUILD_THIRD_PARTY=ON
  make -j
  cd ..
elif [ "$1" = "" ]
then
  mkdir build
  cd ./build
  cmake ..
  make -j
  cd ..
fi