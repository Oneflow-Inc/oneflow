if [ ! -d build ]
then
  mkdir build
  pushd build
  cmake ../ -DCMAKE_BUILD_TYPE=Debug -DUSE_CLANG_FORMAT=ON
  make -j42
  popd
fi

cd build
make -j42
