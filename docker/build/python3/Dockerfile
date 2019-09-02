# warning: never share the container image this dockerfile produces
FROM nvidia/cuda:9.0-cudnn7-devel-centos7
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
    make -j $(nproc) && \
    make install
ENV PATH="/cmake-install/bin:${PATH}"

RUN python3 -m ensurepip
RUN pip3 install numpy protobuf

WORKDIR /workspace/build

COPY cmake /workspace/cmake
COPY CMakeLists.txt /workspace/CMakeLists.txt

# BUILD DEPENDENCY
COPY build/third_party /workspace/build/third_party
RUN cmake -DTHIRD_PARTY=ON -DCMAKE_BUILD_TYPE=Release -DRELEASE_VERSION=ON .. && make -j

# BUILD ONEFLOW
COPY oneflow /workspace/oneflow
COPY tools /workspace/tools

RUN cmake -DTHIRD_PARTY=OFF -DPY3=ON .. && make -j $(nproc) 

## BUILD WHEEL
WORKDIR /workspace
RUN pip3 install wheel
COPY setup.py /workspace/setup.py
RUN python3 setup.py bdist_wheel
RUN pip3 install /workspace/dist/*.whl

RUN rm -rf oneflow third_party cmake CMakeLists.txt
# ENV PYTHONPATH=/workspace/build/python_scripts:$PYTHONPATH
