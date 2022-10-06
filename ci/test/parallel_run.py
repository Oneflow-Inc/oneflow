import asyncio
import os
import argparse
from subprocess import PIPE, STDOUT
import glob
import sys
import time
import socket
from contextlib import closing
import uuid


def gen_cmds(cmd=None, dir=None, doctest=False):
    if doctest:
        paths = glob.glob(os.path.join(dir, "**/*.py"), recursive=True)
        paths = [
            p
            for p in paths
            if "compatible" not in p
            and "single_client" not in p
            and "unittest.py" not in p
        ]
        with_doctest = []
        for p in paths:
            with open(p) as f:
                content = f.read()
                if "import doctest" in content:
                    with_doctest.append("{} {} -v".format(cmd, p))
        print(with_doctest)
        return with_doctest
    else:
        paths = glob.glob(os.path.join(dir, "test_*.py"), recursive=False)
        return ["{} {} --failfast --verbose".format(cmd, p) for p in paths]


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("localhost", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def split_and_print(prefix, text):
    lines = text.splitlines(keepends=True)
    prefixed = ""
    for l in lines:
        prefixed += f"{prefix} {l}"
    print(prefixed, flush=True)


def everyN(l: list, n: int):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def contains_oom_info(txt: str):
    return "memory" in txt or "Memory" in txt or "CUDNN" in txt or "ALLOC" in txt


def should_retry(txt: str):
    return contains_oom_info(txt)


def print_out(prefix: str = "", content: str = ""):
    for l in content.split("\n"):
        print(f"[{prefix}]", l)


async def spawn_shell_and_check(cmd: str = None, gpu_id: int = -1, check: bool = False):
    is_cpu_only = os.getenv("ONEFLOW_TEST_CPU_ONLY")
    print(f"[gpu={gpu_id}]", cmd)
    p = await asyncio.create_subprocess_shell(
        cmd,
        stdout=PIPE,
        stderr=STDOUT,
        env=dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=("-1" if is_cpu_only else ",".join([str(gpu_id)])),
            ONEFLOW_TEST_MASTER_PORT=str(find_free_port()),
            ONEFLOW_TEST_LOG_DIR=("./unittest-log-" + str(uuid.uuid4())),
        ),
    )
    (stdout_data, stderr_data) = await p.communicate()
    decoded = stdout_data.decode()
    if check or should_retry(decoded) == False:
        if p.returncode != 0:
            print_out(prefix=cmd, content=decoded)
            raise RuntimeError(cmd)
    return {"returncode": p.returncode, "cmd": cmd, "stdout": decoded}


async def run_cmds(
    cmds, gpu_num=0, timeout=10, chunk=1, verbose=False, per_gpu_process_num=1
):
    is_cpu_only = os.getenv("ONEFLOW_TEST_CPU_ONLY")
    if is_cpu_only:
        gpu_num = os.cpu_count()
    fails = []
    assert gpu_num > 0
    for cmdN in everyN(cmds, per_gpu_process_num * gpu_num):
        results = await asyncio.gather(
            *[
                spawn_shell_and_check(
                    cmd=cmd, gpu_id=i, check=(per_gpu_process_num == 1)
                )
                for cmd_gpu_num in everyN(cmdN, gpu_num)
                for (i, cmd) in enumerate(cmd_gpu_num)
            ],
        )
        for r in list(results):
            if r["returncode"] != 0:
                fails.append(r["cmd"])
            else:
                print_out(prefix=r["cmd"], content=r["stdout"])
    return fails


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_num", type=int, required=True, default=0)
    parser.add_argument("--dir", type=str, required=True, default=".")
    parser.add_argument("--cmd", type=str, required=False, default=sys.executable)
    parser.add_argument("--timeout", type=int, required=False, default=2)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--verbose", action="store_true", required=False, default=False)
    parser.add_argument("--doctest", action="store_true", required=False, default=False)
    args = parser.parse_args()
    cmds = gen_cmds(cmd=args.cmd, dir=args.dir, doctest=args.doctest)
    start = time.time()
    loop = asyncio.get_event_loop()
    PER_GPU_PROCESS_NUMS = [12, 8, 2, 1]
    is_cpu_only = os.getenv("ONEFLOW_TEST_CPU_ONLY")
    if is_cpu_only:
        PER_GPU_PROCESS_NUMS = [1]
    for per_gpu_process_num in PER_GPU_PROCESS_NUMS:
        print("[per_gpu_process_num]", per_gpu_process_num)
        cmds = loop.run_until_complete(
            run_cmds(
                cmds,
                gpu_num=args.gpu_num,
                timeout=args.timeout,
                chunk=args.chunk,
                verbose=args.verbose,
                per_gpu_process_num=per_gpu_process_num,
            )
        )
    elapsed = time.time() - start
    elapsed_time_txt = time.strftime("elapsed: %H:%M:%S", time.gmtime(elapsed))
    print(elapsed_time_txt)
