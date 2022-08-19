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
import pynvml
import portalocker
import os



def parse_args():
    parser = argparse.ArgumentParser(
        description="Control when the script runs through special variables."
    )
    parser.add_argument(
        "--with-cuda", type=int, default=1, help="whether has cuda device."
    )
    parser.add_argument(
        "cli", type=str, nargs="...", help="the controlled script path."
    )
    return parser.parse_args()


def hash_cli2gpu(cli: str):
    pynvml.nvmlInit()
    slot = pynvml.nvmlDeviceGetCount()
    hash = hashlib.sha1(cli.encode("utf-8")).hexdigest()
    return int(hash, 16) % slot


def main():
    args = parse_args()
    cli = " ".join(args.cli)
    if args.with_cuda:
        gpu_slot = str(hash_cli2gpu(cli))
        with portalocker.Lock(f".oneflow-throttle-gpu-{gpu_slot}.lock", timeout=400):
            cli = "CUDA_VISIBLE_DEVICES=" + gpu_slot + " " + cli
            env = os.environ
            env = dict(env, CUDA_VISIBLE_DEVICES=gpu_slot)
            return subprocess.call(cli, shell=True, env=env)
    else:
        return subprocess.call(cli, shell=True)


if __name__ == "__main__":
    returncode = main()
    exit(returncode)
