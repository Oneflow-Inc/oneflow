import argparse
import hashlib
import subprocess
import pynvml
import portalocker

AUTO_RELEASE_TIME = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control when the script runs through special variables."
    )
    parser.add_argument(
        "--port", type=int, default="6379", help="the redlock server port."
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
        with portalocker.Lock(".oneflow-throttle-gpu" + gpu_slot, timeout=10) as fh:
            cli = "CUDA_VISIBLE_DEVICES=" + gpu_slot + " " + cli
            return subprocess.call(cli, shell=True)
    else:
        return subprocess.call(cli, shell=True)


if __name__ == "__main__":
    returncode = main()
    exit(returncode)
