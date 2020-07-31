set -ex
cd docker/ci/test
docker build --rm \
    -t oneflow-test .
