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
import functools
import os
from typing import Any, Callable, Iterable, List, Optional, Tuple

import torch
import oneflow as flow
import oneflow.support.env_var_util
from oneflow.test_utils.automated_test_util import (
    torch_flow_dual_object as dual_object_module,
)

__all__ = ["profile", "set_profiler_hook", "profile_dual_object", "profiled_framework"]


def compose(*fs):
    def compose2(f, g):
        return lambda *a, **kw: f(g(*a, **kw))

    return functools.reduce(compose2, fs)


class ProfResult:
    def __init__(
        self,
        prof,
        num,
        kind,
        device,
        thread_num,
        op_name,
        args_description,
        additional_description=None,
    ):
        self.prof = prof
        self.num = num
        self.kind = kind
        self.device = device
        self.thread_num = thread_num
        self.op_name = op_name
        self.args_description = args_description
        self.additional_description = additional_description

    def __getattr__(self, attr):
        return getattr(self.prof, attr)


WARMUP_NUM = int(os.getenv("ONEFLOW_PROFILE_WARMUP_NUM", 10))
RUN_NUM = int(os.getenv("ONEFLOW_PROFILE_RUN_NUM", 1000))
PROF_VERBOSE = flow.support.env_var_util.parse_boolean_from_env(
    "ONEFLOW_PROFILE_VERBOSE", False
)
END_TO_END = "end-to-end"


def run_torch(
    op,
    args,
    kwargs,
    device,
    num_threads,
    op_name,
    args_description,
    additional_description=None,
):
    assert device in ["cpu", "cuda"]
    if device == "cpu":
        torch.set_num_threads(num_threads)
        assert torch.get_num_threads() == num_threads
        activities = [torch.profiler.ProfilerActivity.CPU]
    else:
        activities = [torch.profiler.ProfilerActivity.CUDA]

    def tensor_to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    args = [tensor_to_device(arg) for arg in args]
    kwargs = {k: tensor_to_device(v) for k, v in kwargs.items()}
    for _ in range(WARMUP_NUM):
        op(*args, **kwargs)

    if PROF_VERBOSE:
        print(
            f'PyTorch ({f"CPU, num_threads={num_threads}" if device == "cpu" else "GPU"}):'
        )
    with torch.profiler.profile(activities=activities) as prof:
        with torch.profiler.record_function(END_TO_END):
            for _ in range(RUN_NUM):
                op(*args, **kwargs)

    if PROF_VERBOSE:
        print(prof.key_averages().table(row_limit=10))
    return ProfResult(
        prof,
        RUN_NUM,
        "PyTorch",
        device,
        num_threads,
        op_name,
        args_description,
        additional_description,
    )


def run_flow(
    op,
    args,
    kwargs,
    device,
    num_threads,
    op_name,
    args_description,
    additional_description=None,
):
    assert device in ["cpu", "cuda"]
    if device == "cpu":
        # NOTE: there is no flow.get_num_threads()
        flow.set_num_threads(num_threads)
        activities = [flow.profiler.ProfilerActivity.CPU]
    else:
        activities = [flow.profiler.ProfilerActivity.CUDA]

    def tensor_to_device(x):
        if isinstance(x, flow.Tensor):
            return x.to(device)
        return x

    args = [tensor_to_device(arg) for arg in args]
    kwargs = {k: tensor_to_device(v) for k, v in kwargs.items()}
    for _ in range(WARMUP_NUM):
        op(*args, **kwargs)

    if PROF_VERBOSE:
        print(
            f'OneFlow ({f"CPU, num_threads={num_threads}" if device == "cpu" else "GPU"}):'
        )
    with flow.profiler.profile(
        activities=activities,
        record_bandwidth_for_cuda=flow.profiler.ProfilerActivity.CUDA in activities,
    ) as prof:
        with flow.profiler.record_function(END_TO_END):
            for _ in range(RUN_NUM):
                op(*args, **kwargs)

    if PROF_VERBOSE:
        print(prof.key_averages())
    return ProfResult(
        prof,
        RUN_NUM,
        "OneFlow",
        device,
        num_threads,
        op_name,
        args_description,
        additional_description,
    )


def profile_dual_object(op):
    assert isinstance(op, dual_object_module.DualObject)
    torch_op = op.pytorch
    flow_op = op.oneflow

    def profiled_op(*args, **kwargs):
        if "profile_description" in kwargs:
            additional_description = kwargs["profile_description"]
            del kwargs["profile_description"]
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
        for hardware_info in _hardware_info_list:
            if "oneflow" in profiled_framework:
                result.append(
                    run_flow(
                        flow_op,
                        flow_args,
                        flow_kwargs,
                        *hardware_info,
                        op_name,
                        args_description,
                        additional_description,
                    )
                )
            else:
                result.append(None)
        for hardware_info in _hardware_info_list:
            if "pytorch" in profiled_framework:
                result.append(
                    run_torch(
                        torch_op,
                        torch_args,
                        torch_kwargs,
                        *hardware_info,
                        op_name,
                        args_description,
                        additional_description,
                    )
                )
            else:
                result.append(None)
        return _profiler_hook(result)

    return profiled_op


HardwareInfo = Tuple[str, Optional[int]]  # (device_type, num_threads)
_hardware_info_list: List[HardwareInfo] = [("cpu", 1), ("cuda", None)]
_profiler_hook: Callable[[List[ProfResult]], Any] = lambda x: x
profiled_framework: List[str] = ["oneflow", "pytorch"]


def set_hardware_info_list(hardware_info_list: List[HardwareInfo]) -> None:
    global _hardware_info_list
    _hardware_info_list = hardware_info_list


def set_profiler_hook(hook: Callable[[List[ProfResult]], Any]) -> None:
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
