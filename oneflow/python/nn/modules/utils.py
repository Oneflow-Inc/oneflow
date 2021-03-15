"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List

import collections.abc as container_abcs
from itertools import repeat

import oneflow as flow
import oneflow.python.framework.runtime_mode as rt_mode


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


def _list_with_default(out_size, defaults):
    # type: (List[int], List[int]) -> List[int]
    if isinstance(out_size, int):
        return out_size
    if len(defaults) <= len(out_size):
        raise ValueError(
            "Input dimension should be at least {}".format(len(out_size) + 1)
        )
    return [
        v if v is not None else d for v, d in zip(out_size, defaults[-len(out_size) :])
    ]


def _wrapper(func):
    def wrapped_func(*args):
        args = list(args)
        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, flow.Tensor):
                if not arg.is_determined:
                    arg.determine()
                args[i] = arg._local_or_consistent_tensor

        out_list = func(*args)
        for i, out in enumerate(out_list):
            tensor = flow.Tensor(*out.shape)
            tensor._local_or_consistent_tensor = out
            tensor._undetermined_tensor = None
            out_list[i] = tensor

        return out_list

    return wrapped_func


@_wrapper
def op_expr_call(self, *args):
    return self.apply(args)


def global_function_or_identity(*args, **kwargs):
    if rt_mode.CurrentMode() == rt_mode.NORMAL_MODE:
        return flow.global_function(*args, **kwargs)
    else:
        assert rt_mode.CurrentMode() == rt_mode.GLOBAL_MODE
        identity_decorator = lambda func: func
        return identity_decorator
