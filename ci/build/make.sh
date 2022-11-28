set -ex

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
tmp_dir=${ONEFLOW_CI_TMP_DIR:-"$HOME/ci-tmp"}
extra_oneflow_cmake_args=${ONEFLOW_CI_EXTRA_ONEFLOW_CMAKE_ARGS:-""}
package_suffix=${ONEFLOW_CI_PACKAGE_SUFFIX:-""}
cuda_version=${ONEFLOW_CI_CUDA_VERSION:-"10.2"}
python_version_args=${ONEFLOW_CI_PYTHON_VERSION_ARGS:-"--python3.6"}
build_wheel_bash_args=${ONEFLOW_CI_BUILD_WHEEL_BASH_ARGS:-"-l"}
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
    --build-arg from=nvidia/cuda:${cuda_version}-cudnn7-devel-centos7 \
    $docker_proxy_build_args -t $docker_tag .

cd -

# build function
function build() {
    set -x
    docker run --rm \
        -v $tmp_dir:/ci-tmp \
        -w $tmp_dir:/ci-tmp busybox rm -rf /ci-tmp/wheelhouse
    docker run \
        $docker_proxy_run_args \
        --rm $docker_it \
        -v $src_dir:/oneflow-src \
        -v $tmp_dir:/ci-tmp \
        -w /ci-tmp \
        "$docker_tag" \
        bash ${build_wheel_bash_args} /oneflow-src/docker/package/manylinux/build_wheel.sh \
            ${python_version_args} \
            --house-dir /ci-tmp/wheelhouse \
            --package-name oneflow${package_suffix} \
            $extra_oneflow_cmake_args
}

set +e
# reuse cache
build

# clean cache and retry
cached_build_ret=$?
set -e
if [ $cached_build_ret -ne 0 ] && [[ ! -t 1 ]]; then
    echo "retry after cleaning build dir"
    docker run --rm -v $tmp_dir:/ci-tmp busybox sh -c "rm -rf /ci-tmp/*"
    build
fi
