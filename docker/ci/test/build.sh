set -ex
cd docker/ci/test
docker build --rm \
    --build-arg http_proxy=${ONEFLOW_CI_HTTP_PROXY} --build-arg https_proxy=${ONEFLOW_CI_HTTPS_PROXY} \
    -t oneflow-test .
