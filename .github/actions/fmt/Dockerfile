FROM ghcr.io/oneflow-inc/oneflow-fmt:latest
RUN python3 -m pip install black==19.10b0
COPY entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
