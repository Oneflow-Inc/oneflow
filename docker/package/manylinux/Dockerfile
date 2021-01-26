ARG from
FROM ${from}
ARG use_tuna_yum=0
ARG pip_args=""
ARG bazel_url="https://github.com/bazelbuild/bazel/releases/download/3.4.1/bazel-3.4.1-linux-x86_64"
LABEL maintainer="OneFlow Maintainers"

COPY docker/package/manylinux/CentOS-Base.repo /tmp/CentOS-Base.repo
RUN if [ "${use_tuna_yum}" = "1" ]; then mv /tmp/CentOS-Base.repo /etc/yum.repos.d/ && yum makecache ; fi
# to speed up docker img building disable cuda repo
# in 10.1, cuda yum repo will update cublas to 10.2 and breaks build
RUN yum-config-manager --disable cuda nvidia-ml

RUN yum -y install centos-release-scl && \
    yum install -y nasm rdma-core-devel

# overwrite patchelf to fix "maximum size exceed" problem
RUN mkdir -p /tmp && curl -L -o 0.11.zip https://github.com/NixOS/patchelf/archive/0.11.zip && unzip 0.11.zip && cd patchelf-0.11 && sed -i 's/32/64/g' src/patchelf.cc && ./bootstrap.sh && ./configure && make -j`nproc` && make install && cd .. && rm -rf patchelf-0.11 0.11.zip

COPY dev-requirements.txt /tmp/dev-requirements.txt
RUN /opt/python/cp36-cp36m/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && /opt/python/cp37-cp37m/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && /opt/python/cp38-cp38/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && rm /tmp/dev-requirements.txt

RUN curl -L $bazel_url -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel \
    && bazel
