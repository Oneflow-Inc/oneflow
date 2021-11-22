set -ex
conda activate oneflow-dev-clang10-v2
mkdir -p build
cd build
cmake .. -C ../cmake/caches/cn/fast/cpu-clang.cmake
cmake --build . -j $(nproc)
cd -
cd python
python setup.py bdist_wheel
echo "wheelhouse_dir=$PWD/dist" >> $GITHUB_ENV
