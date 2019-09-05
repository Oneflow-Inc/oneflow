docker build \
  --build-arg USE_PYTHON_3_OR_2=2 \
  --build-arg CUDA=9 \
  -t oneflow-build -f docker/build/python2/Dockerfile .
