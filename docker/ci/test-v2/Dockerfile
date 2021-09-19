FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
COPY sources.list /etc/apt/sources.list
RUN apt update && apt install -y --no-install-recommends ffmpeg libsm6 libxext6 gdb gcc g++ make
COPY requirements.txt .
RUN python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt
