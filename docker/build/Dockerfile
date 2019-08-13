# warning: never share the container image this dockerfile produces
FROM nvidia/cuda:10.0-cudnn7-devel-centos7
RUN yum update -y && yum install -y epel-release
RUN yum update -y && yum install -y rdma-core-devel \
    openblas-devel \
    nasm \
    cmake3 \
    make \
    git \
    centos-release-scl \
    rh-python36 \
    python36-devel.x86_64 \
    swig
RUN ln -sf /usr/bin/cmake3 /usr/bin/cmake

RUN mkdir -p /tmp/download/cmake-extracted && \
    cd /tmp/download && \
    curl --location https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0.tar.gz --output cmake.tar.gz && \
    tar -xvzf cmake.tar.gz --directory cmake-extracted && \
    cd cmake-extracted/* && \
    mkdir /cmake-install && \
    cmake . -DCMAKE_INSTALL_PREFIX=/cmake-install && \
    make -j 48 && \
    make install
ENV PATH="/cmake-install/bin:${PATH}"

RUN python3 -m ensurepip
RUN pip3 install numpy protobuf

ENV PYTHONPATH=/workspace/build/python_scripts:$PYTHONPATH

WORKDIR /workspace/build

COPY cmake /workspace/cmake
COPY CMakeLists.txt /workspace/CMakeLists.txt

# BUILD DEPENDENCY
# You might want to remove some lines when building third party from scratch
COPY third_party /workspace/third_party 
COPY build/third_party /workspace/build/third_party
COPY build/nccl /workspace/build/nccl
COPY build/opencv /workspace/build/opencv
RUN cmake -DTHIRD_PARTY=ON .. && make -j

# BUILD ONEFLOW
COPY oneflow /workspace/oneflow
COPY tools /workspace/tools

RUN cmake -DTHIRD_PARTY=OFF -DPY3=ON -DRELEASE_VERSION=OFF .. && make -j 48 
# RUN cmake -DTHIRD_PARTY=OFF .. && make -j 48 
