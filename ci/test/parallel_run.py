import os
import argparse
import multiprocessing as mp
from subprocess import TimeoutExpired
import subprocess
import glob
import sys
import time


def gen_cmds(cmd, dir):
    paths = glob.glob(os.path.join(dir, "test_*.py"), recursive=False)
    return ["{} {} --failfast --verbose".format(cmd, p) for p in paths]


def run_cmds(cmds, gpu_num=0, timeout=10, chunk=1, verbose=False):
    "CUDA_VISIBLE_DEVICES"
    if gpu_num > 0:
        procs = {}
        while len(cmds):

            def available_slots():
                return set(range(gpu_num)) - set(procs.keys())

            while len(available_slots()) >= chunk and len(cmds):
                available_gpu_ids = available_slots()
                gpu_ids = []
                for _i in range(chunk):
                    gpu_id = available_gpu_ids.pop()
                    assert (gpu_id in procs) == False
                    gpu_ids.append(gpu_id)
                cmd = cmds.pop()
                cuda_visible_devices = ",".join([str(i) for i in gpu_ids])
                if verbose:
                    print("cuda_visible_devices:", cuda_visible_devices, "cmd:", cmd)
                proc = subprocess.Popen(
                    cmd,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=cuda_visible_devices),
                    shell=True,
                )
                for gpu_id in gpu_ids:
                    procs[gpu_id] = proc

            gpu_ids_to_release = []
            for gpu_id, proc in procs.items():
                try:
                    proc.wait(timeout=timeout)
                    if proc.returncode == 0:
                        gpu_ids_to_release.append(gpu_id)
                    else:
                        for gpu_id, proc in procs.items():
                            proc.kill()
                            proc.wait()
                        raise ValueError("non-zero returncode found, exiting")
                except TimeoutExpired:
                    if len(available_slots()) >= chunk:
                        break
                    else:
                        continue
            for gpu_id_to_release in gpu_ids_to_release:
                procs.pop(gpu_id_to_release)
    else:
        mp.cpu_count()
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
