set -ex
docker run --rm -it \
    -v $HOME/tensorflow:/tensorflow-src \
    -v `pwd`:`pwd` \
    -w `pwd` oneflow:rel-manylinux2014-cuda-10.2 bash
