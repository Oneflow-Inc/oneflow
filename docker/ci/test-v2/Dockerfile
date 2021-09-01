FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
COPY sources.list /etc/apt/sources.list
RUN apt update && apt install ffmpeg libsm6 libxext6 gdb gcc g++ -y --no-install-recommends
COPY requirements.txt .
RUN python3 -m pip install -i https://mirrors.aliyun.com/pypi/simple -r requirements.txt
