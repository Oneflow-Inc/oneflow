import argparse
import os
from os.path import expanduser
parser = argparse.ArgumentParser()
parser.add_argument("--enable", default=False, action="store_true", required=False)
parser.add_argument("--disable", default=False, action="store_true", required=False)
args = parser.parse_args()
oneflow_share =  os.path.join( expanduser('~'), '.local/share/oneflow')
oneflow_tmp = os.path.join(oneflow_share, 'pypaths')

def main():
    if args.enable:
        import oneflow
        torch_env = os.path.join(oneflow.__path__[0], 'oneflow_tmp')
        torch_path = os.path.join(torch_env, 'torch')
        torch_init = os.path.join(torch_path, '__init__.py')
        if not os.path.exists(torch_path):
            os.mkdir(torch_path)
        with open(torch_init, 'w') as f:
            f.write('from oneflow import *\n')
        if not os.path.exists(oneflow_share):
            os.makedirs(oneflow_share)
        with open(oneflow_tmp, 'a') as f:
            f.write(torch_env) # storing torch environments
            f.write('\n')
        print('export PYTHONPATH=' + torch_env + ':$PYTHONPATH') # run "eval $(oneflow-torch --enable)"
    elif args.disable:
        paths = os.environ['PYTHONPATH'].rstrip(':').split(':')
        if os.path.exists(oneflow_tmp):
            with open(oneflow_tmp, 'r') as f:
                for line in f:
                    line = line.strip()
                    paths = [x for x in paths if x != line]
            os.remove(oneflow_tmp)
        path=':'.join(paths)
        print('export PYTHONPATH='+path)

if __name__ == '__main__':
    main()
