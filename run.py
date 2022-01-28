import subprocess
import sys
import os
import re
import argparse

import numpy as np

def log(x):
    print(x, file=f)

def print_and_log(x):
    print(x)
    print(x, file=f)


class NegativeArgAction(argparse.Action):
    def __init__(self, option_strings, dest, env_var_name, **kwargs):
        assert len(option_strings) == 1
        assert '--no-' in option_strings[0]
        dest = dest[3:]
        super(NegativeArgAction, self).__init__(option_strings, dest, nargs=0, default=True, **kwargs)
        self.env_var_name = env_var_name
        os.environ[self.env_var_name] = "True"
 
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, False)
        os.environ[self.env_var_name] = "False"


class PositiveArgAction(argparse.Action):
    def __init__(self, option_strings, dest, env_var_name, **kwargs):
        super(PositiveArgAction, self).__init__(option_strings, dest, nargs=0, default=True, **kwargs)
        self.env_var_name = env_var_name
        os.environ[self.env_var_name] = "False"
 
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, False)
        os.environ[self.env_var_name] = "True"


parser = argparse.ArgumentParser()
parser.add_argument('runs', type=int)
parser.add_argument('bs', type=int)
parser.add_argument('threshold', type=str)
parser.add_argument('iters', type=int)
parser.add_argument('--no-dtr', action=NegativeArgAction, env_var_name="OF_DTR")
parser.add_argument('--no-lr', action=NegativeArgAction, env_var_name="OF_DTR_LR")
parser.add_argument('--no-o-one', action=NegativeArgAction, env_var_name="OF_DTR_O_ONE")
parser.add_argument('--no-ee', action=NegativeArgAction, env_var_name="OF_DTR_EE")
parser.add_argument('--no-allocator', action=NegativeArgAction, env_var_name="OF_DTR_ALLO")
parser.add_argument('--nlr', action=PositiveArgAction, env_var_name="OF_DTR_NLR")
parser.add_argument('--high-conv', action=PositiveArgAction, env_var_name="OF_DTR_HIGH_CONV")
parser.add_argument('--high-add-n', action=PositiveArgAction, env_var_name="OF_DTR_HIGH_ADD_N")
parser.add_argument('--debug-level', type=int, default=0)

unparsed_args = sys.argv[2:]

args = parser.parse_args()

log_name = f'log_bs{args.bs}_{args.debug_level}{int(args.dtr)}{int(args.allocator)}{int(args.ee)}{int(args.lr)}{int(args.o_one)}{int(args.nlr)}{int(args.high_conv)}{int(args.high_add_n)}_ts{args.threshold}'

f = open(log_name, 'w')

print_and_log(args)

env = os.environ
env.update({'CUDA_VISIBLE_DEVICES': '3'})

p = re.compile('avg ([0-9.]+)s')

times = []


for i in range(args.runs):
    print_and_log(f'run {i}')

    res = subprocess.run(["python3", "rn50_dtr.py", *unparsed_args], capture_output=True, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
    res = res.stdout.decode('utf-8')
    log(res)
    log('\n')
    m = p.search(res)
    assert m is not None
    times.append(float(m.group(1)))

print_and_log(times)
print_and_log(f'avg: {np.mean(times)}')
print_and_log(f'std: {np.std(times)}')
