src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}

cd $src_dir/docker/package/manylinux
docker build -t oneflow:ci-manylinux2014-cuda10.2 .

cd $src_dir
docker run --rm -it -v `pwd`:/oneflow-src oneflow:manylinux2014-cuda10.2 --python3.6
