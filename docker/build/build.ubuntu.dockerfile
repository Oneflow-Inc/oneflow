ARG CUDA=10.0
ARG UBUNTU_VERSION=16.04
FROM nvidia/cuda:${CUDA}-cudnn7-devel-ubuntu${UBUNTU_VERSION}

USER 0

RUN apt-get update && \
    apt-get install -y apt-transport-https && \
    apt-get install -y --no-install-recommends \
    curl \
    default-jdk \
    pciutils \
    nasm \
    make \
    git \
    swig \
    gcc \
    g++ \
    libopenblas-dev \
    python3-dev \
    protobuf-compiler \
    cmake


RUN mkdir -p /tmp/download/cmake-extracted && \
    cd /tmp/download && \
    curl --location https://github.com/Kitware/CMake/releases/download/v3.14.0/cmake-3.14.0.tar.gz --output cmake.tar.gz && \
    tar -xvzf cmake.tar.gz --directory cmake-extracted && \
    cd cmake-extracted/* && \
    mkdir /cmake-install && \
    cmake . -DCMAKE_INSTALL_PREFIX=/cmake-install && \
    make -j $(nproc) && \
    make install
ENV PATH="/cmake-install/bin:${PATH}"

# use this to speed up pip install
ENV TUNA_PIP_INSTALL="python3 -m pip --no-cache-dir install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple"

RUN curl https://bootstrap.pypa.io/get-pip.py --output ./get-pip.py \
    && python3 ./get-pip.py \
    && pip3 install numpy protobuf

WORKDIR /workspace/build

COPY cmake /workspace/cmake
COPY CMakeLists.txt /workspace/CMakeLists.txt

# BUILD DEPENDENCY
COPY build/third_party /workspace/build/third_party
RUN cmake -DTHIRD_PARTY=ON -DCMAKE_BUILD_TYPE=Release -DRELEASE_VERSION=ON .. && make -j

# BUILD ONEFLOW
COPY oneflow /workspace/oneflow
COPY tools /workspace/tools

RUN cmake -DTHIRD_PARTY=OFF -DPY3=ON -DBUILD_TESTING=OFF .. && make -j $(nproc)

# BUILD WHEEL
WORKDIR /workspace
RUN pip3 install wheel
COPY setup.py /workspace/setup.py
RUN python3 setup.py bdist_wheel
RUN pip3 install /workspace/dist/*.whl

RUN rm -rf oneflow third_party cmake CMakeLists.txt
