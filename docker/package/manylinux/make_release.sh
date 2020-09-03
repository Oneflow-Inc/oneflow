set -ex

wheelhouse_dir=/oneflow-src/wheelhouse

package_name=oneflow

function release() {
    set -ex
    docker_tag=oneflow:rel-manylinux2014-cuda-$1
    if [ "$1" == "11.0" ]; then
        cudnn_version=8
    else
        cudnn_version=7
    fi
    docker build --build-arg from=nvidia/cuda:$1-cudnn${cudnn_version}-devel-centos7 -f docker/package/manylinux/Dockerfile -t $docker_tag .
    docker run --rm -it -v `pwd`:/oneflow-src -w /oneflow-src $docker_tag \
        /oneflow-src/docker/package/manylinux/build_wheel.sh --cache-dir /oneflow-src/manylinux2014-build-cache-cuda-$1 \
        --house-dir $wheelhouse_dir \
        --package-name ${package_name}_cu`echo $1 | tr -d .`
}

function release_cpu() {
    docker run --rm -it -v `pwd`:/oneflow-src -w /oneflow-src oneflow:rel-manylinux2014-cuda-10.2 \
        /oneflow-src/docker/package/manylinux/build_wheel.sh --cache-dir /oneflow-src/manylinux2014-build-cache-cpu \
        --house-dir $wheelhouse_dir \
        -DBUILD_CUDA=OFF \
        --package-name "${package_name}_cpu"
}

release_cpu
release 11.0
release 10.2
release 10.1
release 10.0
release 9.2
release 9.1
release 9.0
