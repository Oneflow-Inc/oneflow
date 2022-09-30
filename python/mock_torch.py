import argparse
from pathlib import Path
import oneflow
import os

parser = argparse.ArgumentParser()
parser.add_argument("--enable", default=False, action="store_true", required=False)
parser.add_argument("--disable", default=False, action="store_true", required=False)
args = parser.parse_args()

torch_env = Path(oneflow.__path__[0], 'mock_torch')


def main():
    if args.enable:
        print('export PYTHONPATH=' + str(torch_env) + ':$PYTHONPATH')
    elif args.disable:
        paths = os.environ['PYTHONPATH'].rstrip(':').split(':')
        paths = [x for x in paths if x != str(torch_env)]
        path = ':'.join(paths)
        print('export PYTHONPATH='+path)


if __name__ == '__main__':
    main()
