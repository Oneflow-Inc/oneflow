set -ex

src_dir=${ONEFLOW_SRC_DIR:-"$PWD"}
tmp_dir=${ONEFLOW_CI_TMP_DIR:-"$HOME/ci-tmp"}
mkdir -p $tmp_dir
docker_tag=${ONEFLOW_CI_DOCKER_TAG:-"oneflow:ci-manylinux2014-cuda10.2"}

docker_proxy_build_args=""
docker_proxy_build_args+="--build-arg http_proxy=${ONEFLOW_CI_HTTP_PROXY} --build-arg https_proxy=${ONEFLOW_CI_HTTPS_PROXY}"
docker_proxy_run_args=""
docker_proxy_run_args+="--env http_proxy=${ONEFLOW_CI_HTTP_PROXY} --env https_proxy=${ONEFLOW_CI_HTTPS_PROXY}"

docker run \
    $docker_proxy_run_args \
    -it \
    --rm \
    -v $src_dir:/oneflow-src \
    -v $tmp_dir:/ci-tmp \
    -v $tmp_dir/py-build-lib:/oneflow-src/build/lib/ \
    -w /ci-tmp \
    "$docker_tag" \
    bash
