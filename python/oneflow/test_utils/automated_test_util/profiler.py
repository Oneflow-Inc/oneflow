from typing import Union, List
import functools
from pathlib import Path

import torch
import oneflow as flow
import oneflow.test_utils.automated_test_util.torch_flow_dual_object as dual_object_module
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['profile_torch', 'get_auto_profiler', '_profile', 'profile', 'set_profiler_hook']


def compose(*fs):
    def compose2(f, g):
        return lambda *a, **kw: f(g(*a, **kw))

    return functools.reduce(compose2, fs)


class ProfResult:
    def __init__(self, prof, num, kind, thread_num, op_name, args_description, additional_description=None):
        self.prof = prof
        self.num = num
        self.kind = kind
        self.thread_num = thread_num
        self.op_name = op_name
        self.args_description = args_description
        self.additional_description = additional_description

    def __getattr__(self, attr):
        return getattr(self.prof, attr)


RUN_NUM = 1000

def run_torch(op, args, kwargs, num_threads, op_name, args_description, additional_description=None):
    torch.set_num_threads(num_threads)
    for _ in range(10):
        op(*args, **kwargs)

    print(f'torch (num_threads={torch.get_num_threads()}):')
    with torch.profiler.profile() as prof:
        with torch.profiler.record_function("total"):
            for _ in range(RUN_NUM):
                op(*args, **kwargs)

    print(prof.key_averages().table(row_limit=10))
    return ProfResult(prof, RUN_NUM, 'torch', torch.get_num_threads(), op_name, args_description, additional_description)


def run_flow(op, args, kwargs, num_threads, op_name, args_description, additional_description=None):
    flow.set_num_threads(num_threads)
    for _ in range(10):
        op(*args, **kwargs)

    # NOTE: there is no flow.get_num_threads()
    print(f'flow (num_threads={num_threads}):')
    with flow.profiler.profile() as prof:
        with flow.profiler.record_function("total"):
            for _ in range(RUN_NUM):
                op(*args, **kwargs)

    print(prof.key_averages())
    return ProfResult(prof, RUN_NUM, 'flow', num_threads, op_name, args_description, additional_description)


def profile_dual_object(op):
    assert isinstance(op, dual_object_module.DualObject)
    torch_op = op.pytorch
    flow_op = op.oneflow
    def profiled_op(*args, **kwargs):
        if 'profile_description' in kwargs:
            additional_description = kwargs['profile_description']
            del kwargs['profile_description']
        else:
            additional_description = None

        (
            torch_args,
            torch_kwargs,
            flow_args,
            flow_kwargs,
        ) = dual_object_module.get_args(torch_op, *args, **kwargs)

        op_name = dual_object_module.to_string(op)
        args_description = dual_object_module.to_string(*args, **kwargs)

        result = []
        result.append(run_flow(flow_op, flow_args, flow_kwargs, 32, op_name, args_description, additional_description))
        result.append(run_flow(flow_op, flow_args, flow_kwargs, 8, op_name, args_description, additional_description))
        result.append(run_flow(flow_op, flow_args, flow_kwargs, 1, op_name, args_description, additional_description))
        result.append(run_torch(torch_op, torch_args, torch_kwargs, 32, op_name, args_description, additional_description))
        result.append(run_torch(torch_op, torch_args, torch_kwargs, 8, op_name, args_description, additional_description))
        result.append(run_torch(torch_op, torch_args, torch_kwargs, 1, op_name, args_description, additional_description))
        return _profiler_hook(result)

    return profiled_op


def profile_flow(op):
    def profiled_op(*args, **kwargs):
        return run_flow(op, args, kwargs, 32, None, None)
    return profiled_op


def _profile(op):
    if isinstance(op, dual_object_module.DualObject):
        return profile_dual_object(op)
    else:
        return profile_flow(op)


def add_hook(hook):
    def decorator(op):
        def new_op(*args, **kwargs):
            res = op(*args, **kwargs)
            hook(res)
            return res
        return new_op
    return decorator


profiler_hook = lambda x: x


def set_profiler_hook(hook):
    global _profiler_hook
    _profiler_hook = hook


def profile(op):
    def deco(f):
        def new_f(*args, **kwargs):
            dual_object_module.profiled_method_name.append(op.name)
            res = f(*args, **kwargs)
            dual_object_module.profiled_method_name.pop()
            return res
        return new_f
    return deco


# NOTE:
# What get_auto_profiler does is like applying multiple decorators to a function:
# @add_hook(hook)
# @_profile
# def f(...):
#    ...
# Every decorator has type T -> T,
# so decorator composition is indeed function composition
def get_auto_profiler(hook):
    return compose(add_hook(hook), _profile)


def profile_torch(op):
    def profiled_op(*args, **kwargs):
        return run_torch(op, args, kwargs, 32, None, None)
    return profiled_op


def draw_picture(path: Union[Path, str], label: str, profs: List[ProfResult]):
    if isinstance(path, str):
        path = Path(path)

    torch_profs = list(filter(lambda x: x.kind == 'torch', profs))
    flow_profs = list(filter(lambda x: x.kind == 'flow', profs))
    torch_profs = sorted(torch_profs, key=lambda x: x.thread_num)
    flow_profs = sorted(flow_profs, key=lambda x: x.thread_num)
    i = 0
    labels = []
    torch_times = []
    flow_times = []
    for torch_prof in torch_profs:
        if torch_prof.thread_num == flow_profs[i].thread_num:
            i += 1
            labels.append(f'{torch_prof.thread_num} threads')
            torch_times.append(torch_prof.total_time)


    labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    men_means = [20, 34, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, men_means, width, label='Men')
    rects2 = ax.bar(x + width/2, women_means, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time')
    ax.set_title('Op Time')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()





