"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import argparse
import hashlib
import subprocess
import portalocker
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control when the script runs through special variables."
    )
    parser.add_argument(
        "--with-cuda", type=int, default=1, help="whether has cuda device."
    )
    parser.add_argument("cmd", type=str, nargs="...", help="command to run")
    return parser.parse_args()


def hash_cli2gpu(cmd: list):
    import pynvml

    pynvml.nvmlInit()
    slot = pynvml.nvmlDeviceGetCount()
    hash = hashlib.sha1(" ".join(cmd).encode("utf-8")).hexdigest()
    gpu_id = int(hash, 16) % slot
    return [gpu_id]


def main():
    args = parse_args()
    if args.with_cuda:
        cuda_visible_devices = [str(i) for i in hash_cli2gpu(args.cmd)]
        with portalocker.Lock(
            ".oneflow-throttle-gpu-" + "-".join(cuda_visible_devices) + ".lock",
            timeout=400,
        ):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(cuda_visible_devices))
            return subprocess.call(args.cmd, env=env)
    else:
        return subprocess.call(args.cmd)


if __name__ == "__main__":
    returncode = main()
    exit(returncode)
