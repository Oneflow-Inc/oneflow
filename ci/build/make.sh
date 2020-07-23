set -ex

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
docker_tag=${ONEFLOW_CI_DOCKER_TAG:-"oneflow:ci-manylinux2014-cuda10.2"}

docker_proxy_build_args=""
# docker_proxy_build_args+="--build-arg http_proxy=${ONEFLOW_CI_HTTP_PROXY} --build-arg https_proxy=${ONEFLOW_CI_HTTPS_PROXY}"


cd $src_dir
docker build -f $src_dir/docker/package/manylinux/Dockerfile \
    $docker_proxy_build_args -t $docker_tag .

set +e
cd $src_dir

function build() {
    set -x
    docker run --rm -it -v $src_dir:/oneflow-src "$docker_tag" /oneflow-src/docker/package/manylinux/build_wheel.sh --python3.6
}

build

cached_build_ret=$?

set -e
if [ $cached_build_ret -ne 0 ]; then
    echo "retry after cleaning build dir"
    docker run --rm -it-v $src_dir:/oneflow-src busybox rm -rf /oneflow-src/manylinux2014-build-cache
    build
fi
