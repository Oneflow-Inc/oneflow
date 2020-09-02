set -ex
docker run --rm -it \
    -v $HOME/tensorflow:/tensorflow-src \
    -v `pwd`:/oneflow-src \
    -w /oneflow-src oneflow:rel-manylinux2014-cuda-10.2 bash
