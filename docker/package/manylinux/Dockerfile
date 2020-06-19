FROM nvidia/cuda:10.2-cudnn7-devel-centos7
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

COPY build_scripts /build_scripts
RUN bash build_scripts/build.sh && rm -r build_scripts

ENV SSL_CERT_FILE=/opt/_internal/certs.pem
# manylinux2014 end

RUN yum-config-manager --add-repo https://yum.repos.intel.com/setup/intelproducts.repo && \
    rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB

RUN yum -y install intel-mkl-2020.0-088 
RUN yum -y install nasm swig
RUN /opt/python/cp35-cp35m/bin/pip install -U cmake && ln -s /opt/_internal/cpython-3.5.9/bin/cmake /usr/bin/cmake

# overwrite patchelf to fix "maximum size exceed" problem
RUN curl -L -o 0.11.zip https://github.com/NixOS/patchelf/archive/0.11.zip && unzip 0.11.zip && cd patchelf-0.11 && sed -i 's/32/64/g' src/patchelf.cc && ./bootstrap.sh && ./configure && make -j`nproc` && make install && cd .. && rm -rf patchelf-0.11 0.11.zip

CMD ["/oneflow/docker/package/manylinux/build_wheel.sh"]
