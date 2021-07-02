ARG CUDA=10.0
ARG UBUNTU_VERSION=16.04
FROM nvidia/cuda:${CUDA}-cudnn7-devel-ubuntu${UBUNTU_VERSION}

USER 0

RUN apt-get update && \
    apt-get install -y apt-transport-https && \
    apt-get install -y --no-install-recommends \
    curl \
    nasm \
    make \
    git \
    gcc \
    g++ \
    libopenblas-dev \
    python3-dev

# speed up pip install in China
ENV TUNA_PIP_INSTALL=" -i https://pypi.tuna.tsinghua.edu.cn/simple"

COPY dev-requirements.txt /workspace/dev-requirements.txt

RUN curl https://bootstrap.pypa.io/get-pip.py --output ./get-pip.py \
    && python3 ./get-pip.py \
    && pip3 install $TUNA_INDEX cmake \
    && pip3 install $TUNA_INDEX -r /workspace/dev-requirements.txt

WORKDIR /workspace/build

COPY cmake /workspace/cmake
COPY CMakeLists.txt /workspace/CMakeLists.txt

# BUILD DEPENDENCY
COPY build/third_party /workspace/build/third_party
RUN cmake -DTHIRD_PARTY=ON -DONEFLOW=OFF -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)

# BUILD ONEFLOW
COPY oneflow /workspace/oneflow
COPY tools /workspace/tools

RUN cmake -DTHIRD_PARTY=OFF -DONEFLOW=ON .. && make -j$(nproc) of_pyscript_copy
RUN cmake -DTHIRD_PARTY=OFF -DONEFLOW=ON .. && make -j$(nproc)

# BUILD WHEEL
WORKDIR /workspace
COPY setup.py /workspace/setup.py
RUN python3 setup.py bdist_wheel
RUN pip3 install /workspace/dist/*.whl

RUN rm -rf oneflow third_party cmake CMakeLists.txt
