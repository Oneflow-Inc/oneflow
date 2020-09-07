set -ex

wheelhouse_dir=`pwd`/wheelhouse

package_name=oneflow

tuna_build_args=""
tuna_build_args="--build-arg use_tuna_yum=0 --build-arg pip_args="""

function release() {
    set -ex
    docker_tag=oneflow:rel-manylinux2014-cuda-$1
    if [ "$1" == "11.0" ]; then
        cudnn_version=8
    else
        cudnn_version=7
    fi
    docker build --build-arg from=nvidia/cuda:$1-cudnn${cudnn_version}-devel-centos7 \
        ${tuna_build_args} \
        -f docker/package/manylinux/Dockerfile -t $docker_tag .
    docker run --rm -it -v `pwd`:`pwd` -w `pwd` $docker_tag \
        docker/package/manylinux/build_wheel.sh --cache-dir `pwd`/manylinux2014-build-cache-cuda-$1 \
        --house-dir ${wheelhouse_dir} \
        --package-name ${package_name}_cu`echo $1 | tr -d .`
}

function release_cpu() {
    docker run --rm -it -v `pwd`:`pwd` -w `pwd` oneflow:rel-manylinux2014-cuda-10.2 \
        docker/package/manylinux/build_wheel.sh --cache-dir `pwd`/manylinux2014-build-cache-cpu \
        --house-dir ${wheelhouse_dir} \
        -DBUILD_CUDA=OFF \
        --package-name "${package_name}_cpu"
}

function release_xla() {
    set -ex
    docker_tag=oneflow:rel-manylinux2014-cuda-$1
    if [ "$1" == "11.0" ]; then
        cudnn_version=8
    else
        cudnn_version=7
    fi
    docker build --build-arg from=nvidia/cuda:$1-cudnn${cudnn_version}-devel-centos7 \
        ${tuna_build_args} \
        -f docker/package/manylinux/Dockerfile -t $docker_tag .
    docker run --rm -it -v `pwd`:`pwd` -w `pwd` $docker_tag \
        bash -l docker/package/manylinux/build_wheel.sh --cache-dir `pwd`/manylinux2014-build-cache-cuda-$1-xla \
        --house-dir ${wheelhouse_dir} \
        --package-name ${package_name}_cu`echo $1 | tr -d .`_xla \
        -DWITH_XLA=ON
}

release 11.0
release 10.2
release 10.1
release 10.0
release 9.2
release 9.1
release 9.0

release_cpu

release_xla 11.0
release_xla 10.2
release_xla 10.1
release_xla 10.0
# failed to build XLA with CUDA 9.X
