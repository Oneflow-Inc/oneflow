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
    return ["{} {}".format(cmd, p) for p in paths]


def run_cmds(cmds, gpu_num=0, timeout=10):
    "CUDA_VISIBLE_DEVICES"
    if gpu_num > 0:
        procs = {}
        # gpu_occupant_status = [False] * gpu_num
        while len(cmds):

            def available_slots():
                return set(range(gpu_num)) - set(procs.keys())

            while len(available_slots()) and len(cmds):
                gpu_id = available_slots().pop()
                assert (gpu_id in procs) == False
                cmd = cmds.pop()
                proc = subprocess.Popen(
                    cmd,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id)),
                    shell=True,
                )
                procs[gpu_id] = proc

            gpu_ids_to_release = []
            for gpu_id, proc in procs.items():
                try:
                    proc.wait(timeout=timeout)
                    if proc.returncode == 0:
                        # print("releasing gpu:", gpu_id)
                        gpu_ids_to_release.append(gpu_id)
                    else:
                        for gpu_id, proc in procs.items:
                            proc.kill()
                            proc.wait()
                        exit(-1)
                except TimeoutExpired:
                    if len(available_slots()):
                        break
                    else:
                        continue
            for gpu_id_to_release in gpu_ids_to_release:
                procs.pop(gpu_id_to_release)
    else:
        mp.cpu_count()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_num", type=int, required=True, default=0)
    parser.add_argument("--dir", type=str, required=True, default=".")
    parser.add_argument("--cmd", type=str, required=False, default=sys.executable)
    parser.add_argument("--timeout", type=int, required=False, default=2)
    args = parser.parse_args()
    cmds = gen_cmds(args.cmd, args.dir)
    start = time.time()
    run_cmds(cmds, gpu_num=args.gpu_num, timeout=args.timeout)
    elapsed = time.time() - start
    time.strftime("elapsed: %H:%M:%S", time.gmtime(elapsed))
