docker build \
  --rm \
  --build-arg http_proxy=http://192.168.1.11:8118 \
  --build-arg https_proxy=https://192.168.1.11:8118 \
  -t oneflow-build:ubuntu -f docker/build/build.ubuntu.dockerfile .
