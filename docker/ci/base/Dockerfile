# warning: never share the container image this dockerfile produces
ARG CUDA=10.0

FROM nvidia/cuda:${CUDA}-cudnn7-devel-centos7

COPY dev-requirements.txt /workspace/dev-requirements.txt
RUN yum-config-manager --add-repo https://yum.repos.intel.com/setup/intelproducts.repo && \
    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    yum update -y && yum install -y epel-release && \
    yum update -y && yum install -y rdma-core-devel \
    nasm \
    make \
    git \
    centos-release-scl \
    intel-mkl-2020.0-088 \
    zlib-devel \
    curl-devel \
    which \
    rh-python36 python36-devel.x86_64 python36-devel && \
    python3 -m ensurepip && \
    pip3 install -r /workspace/dev-requirements.txt && \
    yum clean all

RUN mkdir -p /tmp/download && \
    mkdir /cmake-extracted && \
    cd /tmp/download && \
    curl --location https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0-Linux-x86_64.tar.gz --output cmake.tar.gz && \
    tar -xvzf cmake.tar.gz --directory /cmake-extracted && \
    mv /cmake-extracted/* /cmake-extracted/cmake-install && \
    rm -rf /tmp/download

ENV PATH="/cmake-extracted/cmake-install/bin:${PATH}"
