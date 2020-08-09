set -ex

function release() {
    set -ex
    docker_tag=oneflow:rel-manylinux2014-cuda-$1
    package_name=oneflow_cu`echo $1 | tr -d .`
    docker build --build-arg from=nvidia/cuda:$1-cudnn7-devel-centos7 -f docker/package/manylinux/Dockerfile -t $docker_tag .
    docker run --rm -it -v `pwd`:/oneflow-src -w /oneflow-src $docker_tag \
        /oneflow-src/docker/package/manylinux/build_wheel.sh --cache-dir /oneflow-src/manylinux2014-build-cache-cuda-$1 \
        --house-dir /oneflow-src/wheelhouse \
        --package-name $package_name
}

release 10.2
release 10.1
release 10.0
release 9.2
release 9.1
release 9.0
