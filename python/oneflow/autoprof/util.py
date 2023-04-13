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
from typing import Iterable, Union, TypeVar

from rich import box
from rich.console import Console
from rich.table import Table

import csv
import oneflow.test_utils.automated_test_util.profiler as auto_profiler


T = TypeVar("T")


def get_sole_value(x: Iterable[T]) -> T:
    s = set(x)
    assert len(s) == 1
    return list(s)[0]


def get_pytorch_cpu_kernel_time(prof) -> Union[str, float]:
    assert prof.num > 1
    cpu_kernel_items = list(filter(lambda x: x.count >= prof.num, prof.key_averages()))
    if len(cpu_kernel_items) == 0:
        return "-"
    kernel_cpu_time = (
        sum(map(lambda x: x.self_cpu_time_total, cpu_kernel_items)) / prof.num
    )
    return round(kernel_cpu_time, 1)


def get_oneflow_cpu_kernel_time(prof) -> Union[str, float]:
    assert prof.num > 1
    cpu_kernel_items = list(filter(lambda x: x.count >= prof.num, prof.key_averages()))
    if len(cpu_kernel_items) == 0:
        return "-"
    kernel_cpu_time = sum(map(lambda x: x.cpu_time_total, cpu_kernel_items)) / prof.num
    return round(kernel_cpu_time, 1)


def get_pytorch_gpu_kernel_time(prof) -> Union[str, float]:
    gpu_kernel_items = list(filter(lambda x: x.count >= prof.num, prof.key_averages()))
    if len(gpu_kernel_items) == 0:
        return "-"
    kernel_gpu_time = (
        sum(map(lambda x: x.self_cuda_time_total, gpu_kernel_items)) / prof.num
    )
    return round(kernel_gpu_time, 1)


def get_oneflow_gpu_kernel_time(prof) -> Union[str, float]:
    gpu_kernel_items = list(
        filter(lambda x: x.cuda_time_total is not None, prof.key_averages())
    )
    if len(gpu_kernel_items) == 0:
        return "-"
    kernel_gpu_time = sum(map(lambda x: x.cuda_time_total, gpu_kernel_items)) / prof.num
    return round(kernel_gpu_time, 1)


def get_oneflow_gpu_kernel_bandwidth(prof) -> str:
    gpu_kernel_items = list(
        filter(lambda x: x.cuda_time_total is not None, prof.key_averages())
    )
    if len(gpu_kernel_items) == 0:
        return "-"
    if len(gpu_kernel_items) == 1:
        return gpu_kernel_items[0].bandwidth
    return ", ".join([f"{x.name}: {x.bandwidth}" for x in gpu_kernel_items])


def get_pytorch_cpu_end_to_end_time(prof) -> float:
    total = get_sole_value(
        filter(lambda x: x.key == auto_profiler.END_TO_END, prof.key_averages())
    )
    assert total.count == 1
    return round(total.cpu_time / prof.num, 1)


def get_oneflow_cpu_end_to_end_time(prof) -> float:
    total = list(
        filter(lambda x: x.name == auto_profiler.END_TO_END, prof.key_averages())
    )[0]
    assert total.count == 1
    return round(total.cpu_time / prof.num, 1)


def add_row(profs, writer, f):
    non_none_profs = list(filter(lambda x: x is not None, profs))
    op_name = get_sole_value([prof.op_name for prof in non_none_profs])
    args_description = get_sole_value(
        [prof.args_description for prof in non_none_profs]
    )
    additional_description = get_sole_value(
        [prof.additional_description for prof in non_none_profs]
    )
    if "oneflow" in auto_profiler.profiled_framework:
        writer.writerow(
            [
                op_name,
                args_description,
                "OneFlow",
                get_oneflow_gpu_kernel_time(profs[0]),
                get_oneflow_gpu_kernel_bandwidth(profs[0]),
                get_oneflow_cpu_kernel_time(profs[1]),
                get_oneflow_cpu_end_to_end_time(profs[1]),
                get_oneflow_cpu_kernel_time(profs[2]),
                get_oneflow_cpu_end_to_end_time(profs[2]),
                additional_description,
            ]
        )
    if "pytorch" in auto_profiler.profiled_framework:
        writer.writerow(
            [
                op_name,
                args_description,
                "PyTorch",
                get_pytorch_gpu_kernel_time(profs[3]),
                "-",
                get_pytorch_cpu_kernel_time(profs[4]),
                get_pytorch_cpu_end_to_end_time(profs[4]),
                get_pytorch_cpu_kernel_time(profs[5]),
                get_pytorch_cpu_end_to_end_time(profs[5]),
                additional_description,
            ]
        )
    f.flush()


def print_summary_from_csv(filename) -> None:
    print("----------------------------------------------------------------------")
    print(
        'Summary ("KT" means "Kernel Time", "ET" means "End-to-end Time", in microseconds; "BW" means "Bandwidth" in GB/s):'
    )
    with open(filename, "r") as f:
        table = Table(
            "OP",
            "Args",
            "Lib",
            "KT(GPU)",
            "BW(GPU)",
            "KT(1 CPU)",
            "ET(1 CPU)",
            "KT(32 CPU)",
            "ET(32 CPU)",
            box=box.SIMPLE,
        )
        for row in list(csv.reader(f))[1:]:
            row[2] = {"PyTorch": "PT", "OneFlow": "OF"}[row[2]]
            table.add_row(*row[:-1])
        Console().print(table)
