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

ARG USE_PYTHON_3_OR_2=3

RUN if [ "${USE_PYTHON_3_OR_2}" -eq 2 ] ; then yum update -y \
    && yum install -y python-devel.x86_64 \
    && curl https://bootstrap.pypa.io/get-pip.py --output ./get-pip.py \
    && python ./get-pip.py \
    && rm get-pip.py \
    && pip install numpy==1.12.0 protobuf ; fi

COPY dev-requirements.txt /workspace/dev-requirements.txt

RUN if [ "${USE_PYTHON_3_OR_2}" -eq 3 ] ; then yum update -y \
    && yum install -y rh-python36 python36-devel.x86_64 python36-devel \
    && python3 -m ensurepip \
    && pip3 install /workspace/dev-requirements.txt; fi

WORKDIR /workspace/build

COPY cmake /workspace/cmake
COPY CMakeLists.txt /workspace/CMakeLists.txt

# BUILD DEPENDENCY
COPY build/third_party /workspace/build/third_party
RUN cmake -DTHIRD_PARTY=ON -DCMAKE_BUILD_TYPE=Release -DRELEASE_VERSION=ON .. && make -j

# BUILD ONEFLOW
COPY oneflow /workspace/oneflow
COPY tools /workspace/tools

RUN export LD_LIBRARY_PATH=/opt/intel/lib/intel64_lin:/opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH; \
    cmake -DTHIRD_PARTY=OFF .. && make -j $(nproc) ;

## BUILD WHEEL
WORKDIR /workspace
RUN pip${USE_PYTHON_3_OR_2} install wheel
COPY setup.py /workspace/setup.py
RUN python${USE_PYTHON_3_OR_2} setup.py bdist_wheel
RUN pip${USE_PYTHON_3_OR_2} install /workspace/dist/*.whl

RUN rm -rf oneflow third_party cmake CMakeLists.txt
