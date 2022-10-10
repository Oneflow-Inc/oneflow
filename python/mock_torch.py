import argparse
from pathlib import Path
import oneflow
import os

parser = argparse.ArgumentParser()
parser.add_argument('mock', choices=[
                    'enable', 'disable'], help="enable/disable mocking 'import torch', default is enable", nargs='?', default='enable')
args = parser.parse_args()

torch_env = Path(oneflow.__path__[0], 'mock_torch')


def main():
    if args.mock == 'enable':
        print('export PYTHONPATH=' + str(torch_env) + ':$PYTHONPATH')
    elif args.mock == 'disable':
        paths = os.environ['PYTHONPATH'].rstrip(':').split(':')
        paths = [x for x in paths if x != str(torch_env)]
        path = ':'.join(paths)
        print('export PYTHONPATH='+path)


if __name__ == '__main__':
    main()
