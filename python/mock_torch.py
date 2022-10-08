import argparse
from pathlib import Path
import oneflow
import os
from _pytest.monkeypatch import MonkeyPatch
from inspect import getmembers, ismodule
import pkgutil

parser = argparse.ArgumentParser()
parser.add_argument('mock', choices=[
                    'enable', 'disable'], help='enable/disable mocking \'import torch\'')
args = parser.parse_args()

torch_env = Path(oneflow.__path__[0], 'mock_torch')


def patch_torch():
    import torch

    def raiseError(func_name):
        def impl(*args, **kwargs):
            raise NotImplementedError('Oneflow does not define ' + func_name)
        return impl

    def patch_torch_module(mod, mp):
        for k, v in getmembers(mod, lambda x: not ismodule(x)):
            if k.find('__') != -1:
                continue
            of_hasattr = True
            if mod.__name__ == 'torch':
                of_mod = oneflow
            else:
                submod = mod.__name__[len('torch.'):]
                if hasattr(oneflow, submod):
                    of_mod = getattr(oneflow, submod)
                else:
                    of_hasattr = False
            of_attr = k
            if of_hasattr and hasattr(of_mod, of_attr):
                mp.setattr(mod, k, getattr(of_mod, of_attr))
            else:
                print(of_attr)
                mp.setattr(mod, k, raiseError(k))
    mp = MonkeyPatch()
    patch_torch_module(torch, mp)  # top-level module
    for mod in (pkgutil.iter_modules(torch.__path__)):  # sub modules
        if mod.name.startswith('_'):
            continue
        if mod.name == 'contrib':
            continue
        if mod.name == 'fx':
            continue
        if mod.name == 'monitor':
            continue
        if mod.name == 'nested':
            continue
        patch_torch_module(getattr(torch, mod.name), mp)


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
