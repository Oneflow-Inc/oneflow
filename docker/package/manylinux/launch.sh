set -ex
docker run --rm -it \
    -v `pwd`:`pwd` \
    -w `pwd` oneflow:rel-manylinux2014-cuda-11.0 bash
