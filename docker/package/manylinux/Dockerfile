ARG from
FROM ${from}
ARG use_tuna_yum=0
ARG pip_args=""
ARG bazel_url="https://github.com/bazelbuild/bazel/releases/download/3.4.1/bazel-3.4.1-linux-x86_64"
LABEL maintainer="OneFlow Maintainers"

# manylinux2014
ENV AUDITWHEEL_ARCH x86_64
ENV AUDITWHEEL_PLAT manylinux2014_$AUDITWHEEL_ARCH
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV PATH $PATH:/usr/local/bin
ENV LD_LIBRARY_PATH /usr/local/lib64:/usr/local/lib
ENV PKG_CONFIG_PATH /usr/local/lib/pkgconfig

# use tuna mirror
COPY docker/package/manylinux/CentOS7-Base-163.repo /tmp/CentOS-Base.repo
RUN if [ "${use_tuna_yum}" = "1" ]; then mv /tmp/CentOS-Base.repo /etc/yum.repos.d/ && yum makecache ; fi

# to speed up docker img building disable cuda repo
# in 10.1, cuda yum repo will update cublas to 10.2 and breaks build
RUN yum-config-manager --disable cuda nvidia-ml

ARG MANYLINUX_SHA=b634044
RUN yum -y install unzip && curl -L -o manylinux.zip https://github.com/Oneflow-Inc/manylinux/archive/${MANYLINUX_SHA}.zip && unzip manylinux.zip -d tmp && cp -r tmp/*/docker/build_scripts /build_scripts && bash build_scripts/build.sh && rm -r build_scripts tmp manylinux.zip

ENV SSL_CERT_FILE=/opt/_internal/certs.pem
# manylinux2014 end

RUN yum-config-manager --add-repo https://yum.repos.intel.com/oneapi && \
    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    yum update -y && yum install -y epel-release && \
    yum -y install centos-release-scl && \
    yum install -y intel-oneapi-mkl-devel-2021.2.0 nasm rdma-core-devel devtoolset-7-gcc* rsync gdb

RUN /opt/python/cp35-cp35m/bin/pip install $pip_args -U cmake==3.18.4.post1 && ln -s /opt/_internal/cpython-3.5.9/bin/cmake /usr/bin/cmake

RUN mkdir -p /tmp && cd /tmp && \
    curl -L -o patchelf-src.zip \
    https://github.com/Oneflow-Inc/patchelf/archive/64bf5388ef7d45d3697c4aadbd3f5d7d68a22aa3.zip && \
    unzip patchelf-src.zip && cd patchelf-* && ./bootstrap.sh && ./configure && make -j`nproc` && \
    make install && cd .. && rm -rf patchelf-*

RUN curl -L $bazel_url -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel \
    && bazel

COPY dev-requirements.txt /tmp/dev-requirements.txt
RUN /opt/python/cp36-cp36m/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && /opt/python/cp37-cp37m/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && /opt/python/cp38-cp38/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && rm /tmp/dev-requirements.txt
