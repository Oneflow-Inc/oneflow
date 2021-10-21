FROM ufoym/deepo

RUN apt remove openmpi-common libfabric1 openmpi-bin librdmacm1:amd64 libopenmpi2 libopenmpi2:amd64 -y
ENV MOFED_DIR MLNX_OFED_LINUX-4.3-1.0.1.0-ubuntu18.04-x86_64
RUN wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/deps/${MOFED_DIR}.tgz && \
    tar -xzvf ${MOFED_DIR}.tgz && \
    ${MOFED_DIR}/mlnxofedinstall --user-space-only --without-fw-update --all -q --force && \
    cd .. && \
    rm -rf ${MOFED_DIR} && \
    rm -rf *.tgz

RUN apt update && apt install -y --no-install-recommends gdb openssh-server openssh-client

RUN echo 'ALL ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

COPY requirements.txt .
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt
