set -ex

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
tmp_dir=${ONEFLOW_CI_TMP_DIR:-"$HOME/ci-tmp"}
mkdir -p $tmp_dir
docker_tag=${ONEFLOW_CI_DOCKER_TAG:-"oneflow:ci-manylinux2014-cuda10.2"}

docker_proxy_build_args=""
docker_proxy_build_args+="--build-arg http_proxy=${ONEFLOW_CI_HTTP_PROXY} --build-arg https_proxy=${ONEFLOW_CI_HTTPS_PROXY}"
docker_proxy_run_args=""
docker_proxy_run_args+="--env http_proxy=${ONEFLOW_CI_HTTP_PROXY} --env https_proxy=${ONEFLOW_CI_HTTPS_PROXY}"

docker_it=""
if [[ -t 1 ]]; then
    docker_it="-it"
fi

# build manylinux image
cd $src_dir
docker build -f $src_dir/docker/package/manylinux/Dockerfile \
    --build-arg from=nvidia/cuda:10.2-cudnn7-devel-centos7 \
    $docker_proxy_build_args -t $docker_tag .

cd -
# build function
function build() {
    set -x
    docker run \
        $docker_proxy_run_args \
        --rm $docker_it \
        -v $src_dir:/oneflow-src \
        -v $tmp_dir:/ci-tmp \
        -v $tmp_dir/py-build-lib:/oneflow-src/build/lib/ \
        -w /ci-tmp \
        "$docker_tag" \
        /oneflow-src/docker/package/manylinux/build_wheel.sh \
            --python3.6 \
            --package-name oneflow_cu102
}

set +e
# reuse cache
build

# clean cache and retry
cached_build_ret=$?
set -e
if [ $cached_build_ret -ne 0 ] && [[ ! -t 1 ]]; then
    echo "retry after cleaning build dir"
    docker run --rm -v $tmp_dir:/ci-tmp busybox rm -rf /ci-tmp/*
    build
fi
