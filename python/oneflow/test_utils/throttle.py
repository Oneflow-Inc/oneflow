import argparse
import hashlib
import subprocess
import pynvml
from redis import Redis
from pottery import Redlock

AUTO_RELEASE_TIME = 300

def parse_args():
    parser = argparse.ArgumentParser(description='Control when the script runs through special variables.')
    parser.add_argument('--port', type=int, default='6379', help='the redlock server port.')
    parser.add_argument('--with-cuda', type=bool, default=True, help='whether has cuda device.')
    parser.add_argument('cli', type=str, nargs='...', help='the controlled script path.')
    return parser.parse_args()

def get_redis(port:int):
    server_host = 'redis://localhost:'+ str(port)
    return Redis.from_url(server_host)

def hash_cli2gpu(cli:str):
    pynvml.nvmlInit()
    slot = pynvml.nvmlDeviceGetCount()
    hash = hashlib.sha1(cli.encode('utf-8')).hexdigest()
    return int(hash, 16) % slot

def run(cli) -> subprocess.Popen:
    proc = subprocess.Popen(cli, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    while proc.poll() is None:
        line = proc.stdout.readline().decode("utf8")
        print(line, end='')
    proc.wait()
    return proc.returncode


def main():
    args = parse_args()
    cli = ' '.join(args.cli)
    if args.with_cuda:
        redis = get_redis(args.port)
        gpu_slot = str(hash_cli2gpu(cli))
        lock = Redlock(key=gpu_slot, masters={redis}, auto_release_time=AUTO_RELEASE_TIME)
        with lock:
            cli = 'CUDA_VISIBLE_DEVICES='+ gpu_slot + ' ' + cli
            return run(cli)
    else:
        return run(cli)

if __name__ == "__main__":
    exit(main())
