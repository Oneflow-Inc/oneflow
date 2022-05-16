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
"""
This file is mostly copied from PyTorch v1.8.1 torch/distributed/launch.py
"""
import asyncio
import os
import random
import sys
from argparse import REMAINDER, ArgumentParser
from typing import IO, Any, List, Optional
import glob
import hashlib
from math import ceil

stdout_filename = "stdout"
stderr_filename = "stderr"

global PARALLEL_NUM
global SUCCESS_NUM
PARALLEL_NUM = 0
SUCCESS_NUM = 0


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description="helper to start multiple distributed launches in parallel"
    )
    parser.add_argument(
        "--files",
        type=str,
        help="files to run, support pattern",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        help="for one command, how many duplications to run",
        required=True,
    )
    parser.add_argument(
        "--device_num", type=int, help="how many devices to run on", required=True,
    )
    parser.add_argument(
        "-n",
        "--parallel_num",
        type=str,
        help="how many launches, could be a number, or 'master_port'",
        required=True,
    )
    parser.add_argument(
        "--auto_cuda_visible_devices",
        action="store_true",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--shuffle", action="store_true", required=False, default=False,
    )
    parser.add_argument(
        "--verbose", action="store_true", required=False, default=False,
    )
    parser.add_argument(
        "--master_port",
        default=[],
        action="append",
        help="Master node (rank 0)'s free port, pass this multiple `--master_port` to launch more instances",
    )
    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script as a python module, executing with the same behavior as'python -m'.",
    )
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training program/script to be launched in parallel, followed by all the arguments for the training script",
    )
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()


async def run_and_capture(cmd=None, prefix=None, **kwargs):
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, **kwargs
    )
    while True:
        line = await proc.stdout.readline()
        print(prefix, line.decode(), end="")
        if not line:
            break
    await proc.wait()
    assert proc.returncode == 0, prefix
    global PARALLEL_NUM
    global SUCCESS_NUM
    SUCCESS_NUM += 1
    print(f"{prefix} succeed ({SUCCESS_NUM}/{PARALLEL_NUM})")


async def launch_multiple(
    cmds=None, group_size=None, auto_cuda_env=False, device_num=None
):
    visible_groups = [
        [str(x) for x in range(device_num)[i : i + group_size]]  # to get ["0", "1"]
        for i in range(0, device_num, group_size)
    ]
    spawns = []
    for i, cmd in enumerate(cmds):
        group_idx = i % len(visible_groups)
        cuda_visible_devices = ",".join(visible_groups[group_idx])
        print(cuda_visible_devices, cmd, "\n")
        env = os.environ
        if auto_cuda_env:
            env = dict(env, CUDA_VISIBLE_DEVICES=cuda_visible_devices)
        process = run_and_capture(
            cmd=cmd, prefix=f"[wg={i}][device={cuda_visible_devices}]", env=env,
        )
        spawns.append(process)
    await asyncio.gather(*spawns)


def main():
    args = parse_args()
    # find files and chuck them
    files = []
    for f in args.files:
        files += list(glob.glob(f, recursive=True))
    print("total files:", len(files))
    files = sorted(
        files,
        key=lambda x: hashlib.md5(os.path.basename(x.encode("ascii"))).hexdigest(),
    )
    if args.shuffle:
        random.shuffle(files)
    files_hash = hashlib.md5(
        "".join([os.path.basename(x) for x in files]).encode()
    ).hexdigest()[:8]
    if args.verbose:
        print(
            f"::warning file=testFilesHash,line={len(files)},col=0,endColumn=0::shuffle-{args.shuffle}-group_size-{args.group_size}-md5-{files_hash}"
        )
    if args.parallel_num == "master_port":
        parallel_num = len(args.master_port)
        master_ports = args.master_port
    else:
        parallel_num = int(args.parallel_num)
        if parallel_num != len(args.master_port):
            print(
                "warning", "parallel_num != len(args.master_port)", "will auto generate"
            )
        default_master_port = 29500
        master_ports = list(
            range(default_master_port, default_master_port + parallel_num)
        )
    assert parallel_num > 0
    assert len(master_ports) == parallel_num
    chunk_size = ceil(len(files) / parallel_num)
    global PARALLEL_NUM
    PARALLEL_NUM = parallel_num
    chunks = [files[i : i + chunk_size] for i in range(0, len(files), chunk_size)]

    # check args
    assert args.training_script == "oneflow.distributed.launch"

    # generate commands
    cmds = [
        [sys.executable, "-m", args.training_script, "--master_port", str(master_port)]
        + args.training_script_args
        + chunck
        for (master_port, chunck) in zip(master_ports, chunks)
    ]
    loop = asyncio.get_event_loop()
    processes = launch_multiple(
        cmds=cmds,
        auto_cuda_env=args.auto_cuda_visible_devices,
        group_size=args.group_size,
        device_num=args.device_num,
    )
    loop.run_until_complete(processes)


if __name__ == "__main__":
    main()
