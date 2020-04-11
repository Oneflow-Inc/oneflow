# warning: never share the container image this dockerfile produces
ARG CUDA=10.0

FROM nvidia/cuda:${CUDA}-cudnn7-devel-centos7
RUN yum-config-manager --add-repo https://yum.repos.intel.com/setup/intelproducts.repo && \
    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN yum update -y && yum install -y epel-release
RUN yum update -y && yum install -y rdma-core-devel \
    nasm \
    cmake3 \
    make \
    git \
    centos-release-scl \
    swig \
    intel-mkl-2020.0-088 \
    zlib-devel \
    curl-devel \
    which

RUN ln -sf /usr/bin/cmake3 /usr/bin/cmake

RUN mkdir -p /tmp/download/cmake-extracted && \
    cd /tmp/download && \
    curl --location https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0.tar.gz --output cmake.tar.gz && \
    tar -xvzf cmake.tar.gz --directory cmake-extracted && \
    cd cmake-extracted/* && \
    mkdir /cmake-install

RUN cd /tmp/download/cmake-extracted/* && \
    cmake . -DCMAKE_USE_SYSTEM_CURL=ON -DCMAKE_INSTALL_PREFIX=/cmake-install && \
    make -j $(nproc) && \
    make install
ENV PATH="/cmake-install/bin:${PATH}"

RUN yum update -y \
    && yum install -y rh-python36 python36-devel.x86_64 python36-devel \
    && python3 -m ensurepip \
    && pip3 install numpy protobuf  && yum clean all
