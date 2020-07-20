set -ex

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
docker_tag=${ONEFLOW_CI_DOCKER_TAG:-"oneflow:ci-manylinux2014-cuda10.2"}

docker_proxy_build_args=""
docker_proxy_build_args+="--build-arg http_proxy=${ONEFLOW_CI_HTTP_PROXY} --build-arg https_proxy=${ONEFLOW_CI_HTTPS_PROXY}"


cd $src_dir/docker/package/manylinux
docker build $docker_proxy_build_args -t $docker_tag .

set +e
cd $src_dir
docker run --rm -it -v $src_dir:/oneflow-src "$docker_tag" --python3.6

cached_build_ret=$?
set -e
if [ $cached_build_ret -ne 0 ]; then
    rm -rf $src_dir/manylinux2014-build-cache
    docker run --rm -it -v $src_dir:/oneflow-src "$docker_tag" --python3.6
fi
