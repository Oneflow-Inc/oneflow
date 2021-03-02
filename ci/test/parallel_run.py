import os
import argparse
from subprocess import TimeoutExpired
import subprocess
import glob
import sys
import time
import socket
from contextlib import closing
import uuid


def gen_cmds(cmd, dir):
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


def run_cmds(cmds, gpu_num=0, timeout=10, chunk=1, verbose=False):
    "CUDA_VISIBLE_DEVICES"
    if gpu_num > 0:
        proc2gpu_ids = {}
        while len(cmds):

            def available_slots():
                occupied_gpu_ids = set({})
                for _p, gpu_ids in proc2gpu_ids.items():
                    assert isinstance(gpu_ids, list)
                    occupied_gpu_ids.update(gpu_ids)
                return set(range(gpu_num)) - occupied_gpu_ids

            while len(available_slots()) >= chunk and len(cmds):
                available_gpu_ids = available_slots()
                gpu_ids_to_occupy = []
                for _i in range(chunk):
                    gpu_id = available_gpu_ids.pop()
                    gpu_ids_to_occupy.append(gpu_id)
                cmd = cmds.pop()
                cuda_visible_devices = ",".join([str(i) for i in gpu_ids_to_occupy])
                if verbose:
                    print(
                        "cuda_visible_devices:",
                        cuda_visible_devices,
                        "cmd:",
                        cmd,
                        flush=True,
                    )
                proc = subprocess.Popen(
                    cmd,
                    env=dict(
                        os.environ,
                        CUDA_VISIBLE_DEVICES=cuda_visible_devices,
                        ONEFLOW_TEST_MASTER_PORT=str(find_free_port()),
                        ONEFLOW_TEST_LOG_DIR=("./unittest-log-" + str(uuid.uuid4())),
                    ),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    encoding="utf-8",
                    shell=True,
                )
                proc2gpu_ids[proc] = gpu_ids_to_occupy

            procs_to_release = []
            for proc, gpu_ids in proc2gpu_ids.items():
                outs = None
                errs = None
                try:
                    outs, errs = proc.communicate(timeout=1)
                    if outs:
                        split_and_print(f"[{proc.args}][stdout]", outs)
                    if errs:
                        split_and_print(f"[{proc.args}][stderr]", errs)
                    if proc.returncode == 0:
                        procs_to_release.append(proc)
                    else:
                        for proc_to_kill in proc2gpu_ids.keys():
                            proc_to_kill.kill()
                            proc_to_kill.wait()
                        raise ValueError("non-zero returncode found", proc.args)
                except TimeoutExpired:
                    pass
            for proc in procs_to_release:
                proc2gpu_ids.pop(proc)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_num", type=int, required=True, default=0)
    parser.add_argument("--dir", type=str, required=True, default=".")
    parser.add_argument("--cmd", type=str, required=False, default=sys.executable)
    parser.add_argument("--timeout", type=int, required=False, default=2)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--verbose", action="store_true", required=False, default=False)
    args = parser.parse_args()
    cmds = gen_cmds(args.cmd, args.dir)
    start = time.time()
    run_cmds(
        cmds,
        gpu_num=args.gpu_num,
        timeout=args.timeout,
        chunk=args.chunk,
        verbose=args.verbose,
    )
    elapsed = time.time() - start
    elapsed_time_txt = time.strftime("elapsed: %H:%M:%S", time.gmtime(elapsed))
    print(elapsed_time_txt)
