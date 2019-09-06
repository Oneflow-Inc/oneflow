docker build \
  --build-arg USE_PYTHON_3_OR_2=3 \
  --build-arg CUDA=10 \
  -t oneflow-build -f docker/build/Dockerfile --rm .
