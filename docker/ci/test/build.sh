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

img_tag="oneflow-test:0.2" # update me if any of related files are changed
if [[ "$(docker images -q ${img_tag} 2> /dev/null)" == "" ]]; then
  docker build --rm $proxy_args \
    -t $img_tag .
fi
