import os
import argparse
from pathlib import Path
import re
import json
import subprocess


def check_and_download(tag, url):
    img_dir = os.path.join(os.path.expanduser("~"), "imgs")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    returncode = subprocess.run(
        f"docker image inspect {tag}",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    if returncode == 0:
        print("[OK]", tag)
    else:
        basename = os.path.basename(url)
        dst = os.path.join(img_dir, basename)
        subprocess.check_call(f"wget -c {url} -O {dst}", shell=True)
        subprocess.check_call(f"docker load -i {dst}", shell=True)
        base = os.path.basename(dst)
        base = os.path.splitext(base)[0]
        base = os.path.splitext(base)[0]
        keep_tag = f"ofkeep:{base}"
        subprocess.check_call(f"docker tag {tag} {keep_tag}", shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--create_index", action="store_true", required=False, default=False
    )
    args = parser.parse_args()
    imgs = [
        {
            "tag": "nvidia/cuda:10.0-cudnn7-devel-centos7",
            "url": "https://oneflow-static.oss-cn-beijing.aliyuncs.com/img/nvidiacuda10.0-cudnn7-devel-centos7.tar.gz",
        },
        {
            "tag": "nvidia/cuda:10.1-cudnn7-devel-centos7",
            "url": "https://oneflow-static.oss-cn-beijing.aliyuncs.com/img/nvidiacuda10.1-cudnn7-devel-centos7.tar.gz",
        },
        {
            "tag": "nvidia/cuda:10.2-cudnn7-devel-centos7",
            "url": "https://oneflow-static.oss-cn-beijing.aliyuncs.com/img/nvidiacuda10.2-cudnn7-devel-centos7.tar.gz",
        },
        {
            "tag": "nvidia/cuda:11.0-cudnn8-devel-centos7",
            "url": "https://oneflow-static.oss-cn-beijing.aliyuncs.com/img/nvidiacuda11.0-cudnn8-devel-centos7.tar.gz",
        },
        {
            "tag": "nvidia/cuda:11.1-cudnn8-devel-centos7",
            "url": "https://oneflow-static.oss-cn-beijing.aliyuncs.com/img/nvidiacuda11.1-cudnn8-devel-centos7.tar.gz",
        },
    ]
    for img in imgs:
        check_and_download(img["tag"], img["url"])
