set -ex
test_img_dir="$(dirname "${BASH_SOURCE[0]}")"
test_img_dir="$(realpath "${test_img_dir}")"
cd $test_img_dir

proxy_args=""
proxy_args+=" --network=host"
proxy_args+=" --build-arg HTTP_PROXY=${HTTP_PROXY}"
proxy_args+=" --build-arg HTTPS_PROXY=${HTTPS_PROXY}"
proxy_args+=" --build-arg http_proxy=${http_proxy}"
proxy_args+=" --build-arg https_proxy=${https_proxy}"

docker build --rm $proxy_args \
    -t oneflow-test:$USER .
