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
COPY docker/package/manylinux/CentOS-Base.repo /tmp/CentOS-Base.repo

# to speed up docker img building disable cuda repo
# in 10.1, cuda yum repo will update cublas to 10.2 and breaks build
RUN yum-config-manager --disable cuda nvidia-ml

RUN if [ "${use_tuna_yum}" = "1" ]; then mv /tmp/CentOS-Base.repo /etc/yum.repos.d/ && yum makecache ; fi

ARG MANYLINUX_SHA=b634044
RUN yum -y install unzip && curl -L -o manylinux.zip https://github.com/Oneflow-Inc/manylinux/archive/${MANYLINUX_SHA}.zip && unzip manylinux.zip -d tmp && cp -r tmp/*/docker/build_scripts /build_scripts && bash build_scripts/build.sh && rm -r build_scripts tmp manylinux.zip

ENV SSL_CERT_FILE=/opt/_internal/certs.pem
# manylinux2014 end

RUN yum-config-manager --add-repo https://yum.repos.intel.com/setup/intelproducts.repo && \
    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    yum update -y && yum install -y epel-release && \
    yum -y install centos-release-scl && \
    yum install -y intel-mkl-64bit-2020.0-088 nasm rdma-core-devel devtoolset-7-gcc* rsync

RUN /opt/python/cp35-cp35m/bin/pip install $pip_args -U cmake && ln -s /opt/_internal/cpython-3.5.9/bin/cmake /usr/bin/cmake

# overwrite patchelf to fix "maximum size exceed" problem
RUN mkdir -p /tmp && curl -L -o 0.11.zip https://github.com/NixOS/patchelf/archive/0.11.zip && unzip 0.11.zip && cd patchelf-0.11 && sed -i 's/32/64/g' src/patchelf.cc && ./bootstrap.sh && ./configure && make -j`nproc` && make install && cd .. && rm -rf patchelf-0.11 0.11.zip

RUN curl -L $bazel_url -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel \
    && bazel

COPY dev-requirements.txt /tmp/dev-requirements.txt
RUN /opt/python/cp35-cp35m/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && /opt/python/cp36-cp36m/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && /opt/python/cp37-cp37m/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && /opt/python/cp38-cp38/bin/pip install $pip_args -r /tmp/dev-requirements.txt --user \
    && rm /tmp/dev-requirements.txt
