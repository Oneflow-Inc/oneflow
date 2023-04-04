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
import os
import signal
import subprocess
import sys
import time
from argparse import REMAINDER, ArgumentParser
from typing import IO, Any, List, Optional

stdout_filename = "stdout"
stderr_filename = "stderr"


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description="OneFlow distributed training launch helper utility that will spawn up multiple distributed processes"
    )
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="The number of nodes to use for distributed training",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="The rank of the node for multi-node distributed training",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="The number of processes to launch on each node, for GPU training, this is recommended to be set to the number of GPUs in your system so that each process can be bound to a single GPU.",
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
        type=str,
        help="Master node (rank 0)'s address, should be either the IP address or the hostname of node 0, for single node multi-proc training, the --master_addr can simply be 127.0.0.1",
    )
    parser.add_argument(
        "--master_port",
        default=29500,
        type=int,
        help="Master node (rank 0)'s free port that needs to be used for communication during distributed training",
    )
    parser.add_argument(
        "-m",
        "--module",
        default=False,
        action="store_true",
        help="Changes each process to interpret the launch script as a python module, executing with the same behavior as'python -m'.",
    )
    parser.add_argument(
        "--no_python",
        default=False,
        action="store_true",
        help='Do not prepend the training script with "python" - just exec it directly. Useful when the script is not a Python script.',
    )
    parser.add_argument(
        "--redirect_stdout_and_stderr",
        default=False,
        action="store_true",
        help=f"write the stdout and stderr to files\n                    '{stdout_filename}' and '{stderr_filename}' in logdir.",
    )
    parser.add_argument(
        "--logdir",
        default="log",
        type=str,
        help=f"Relative path to write subprocess logs to. Passing in a relative\n        path will create a directory if needed. Note that\n        successive runs with the same path to write logs to will overwrite existing logs,\n        so be sure to save logs as needed.",
    )
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the single GPU training program/script to be launched in parallel, followed by all the arguments for the training script",
    )
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()
    dist_world_size = args.nproc_per_node * args.nnodes
    current_env = os.environ.copy()
    current_env["MASTER_ADDR"] = args.master_addr
    current_env["MASTER_PORT"] = str(args.master_port)
    current_env["WORLD_SIZE"] = str(dist_world_size)

    if args.master_port is None or args.master_port >= 2 ** 16:
        raise ValueError(
            f"The port number of the master endpoint '{args.master_addr}:{args.master_port}' must be an integer "
            "between 0 and 65536."
        )

    if "OMP_NUM_THREADS" not in os.environ and args.nproc_per_node > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        print(
            "*****************************************\n"
            "Setting OMP_NUM_THREADS environment variable for each process "
            "to be {} in default, to avoid your system being overloaded, "
            "please further tune the variable for optimal performance in "
            "your application as needed. \n"
            "*****************************************".format(
                current_env["OMP_NUM_THREADS"]
            )
        )

    processes: List[Any] = []

    if (
        args.redirect_stdout_and_stderr
        and os.path.exists(args.logdir)
        and not os.path.isdir(args.logdir)
    ):
        raise ValueError("argument --logdir must be a path to a directory.")

    subprocess_file_handles = []
    for local_rank in range(0, args.nproc_per_node):
        dist_rank = args.nproc_per_node * args.node_rank + local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)
        with_python = not args.no_python
        cmd = []
        if with_python:
            cmd = [sys.executable, "-u"]
            if args.module:
                cmd.append("-m")
        elif args.module:
            raise ValueError(
                "Don't use both the '--no_python' flag and the '--module' flag at the same time."
            )
        cmd.append(args.training_script)
        cmd.extend(args.training_script_args)
        stdout_handle: Optional[IO]
        stderr_handle: Optional[IO]
        log_directory_path = os.path.join(
            os.getcwd(), args.logdir, f"local_rank_{local_rank}"
        )
        current_env["GLOG_log_dir"] = log_directory_path
        if args.redirect_stdout_and_stderr:
            os.makedirs(log_directory_path, exist_ok=True)
            node_rank = args.node_rank
            stdout_handle = open(os.path.join(log_directory_path, stdout_filename), "w")
            stderr_handle = open(os.path.join(log_directory_path, stderr_filename), "w")
            subprocess_file_handles.append((stdout_handle, stderr_handle))
            stdout_name = stdout_handle.name
            stderr_name = stderr_handle.name
            print(
                f"Note: Stdout and stderr for node {node_rank} rank {local_rank} will\n            be written to {stdout_name}, {stderr_name} respectively."
            )
        sig_names = {2: "SIGINT", 15: "SIGTERM"}
        last_return_code = None

        # set killing flag to make sure killing signal only executed once
        kill_flag = True

        def sigkill_handler(signum, frame):
            nonlocal kill_flag
            if not kill_flag:
                return
            for process in processes:
                print(f"Killing subprocess {process.pid}")
            kill_flag = False
            try:
                # Note: use os.kill or process.kill() may only kill current process
                # use killpg will kill(use signal) this process and all sub-processes
                #
                # Note: Worker processes launched by data loader will exit automatically
                # when its parent process exits because of `_prctl_pr_set_pdeathsig`.
                os.killpg(os.getpid(), signal.SIGTERM)
            except Exception:
                pass
            if last_return_code is not None:
                raise subprocess.CalledProcessError(
                    returncode=last_return_code, cmd=cmd
                )
            if signum in sig_names:
                print(f"Main process received {sig_names[signum]}, exiting")
            sys.exit(1)

        signal.signal(signal.SIGINT, sigkill_handler)
        signal.signal(signal.SIGTERM, sigkill_handler)
        stdout_handle = (
            None
            if not subprocess_file_handles
            else subprocess_file_handles[local_rank][0]
        )
        stderr_handle = (
            None
            if not subprocess_file_handles
            else subprocess_file_handles[local_rank][1]
        )
        process = subprocess.Popen(
            cmd, env=current_env, stdout=stdout_handle, stderr=stderr_handle
        )
        processes.append(process)
    try:
        alive_processes = set(processes)
        while len(alive_processes):
            finished_processes = []
            for process in alive_processes:
                if process.poll() is None:
                    continue
                elif process.returncode != 0:
                    last_return_code = process.returncode
                    sigkill_handler(signal.SIGTERM, None)
                else:
                    finished_processes.append(process)
            alive_processes = set(alive_processes) - set(finished_processes)
            time.sleep(1)
    finally:
        for (stdout_handle, stderr_handle) in subprocess_file_handles:
            stdout_handle.close()
            stderr_handle.close()


if __name__ == "__main__":
    main()
