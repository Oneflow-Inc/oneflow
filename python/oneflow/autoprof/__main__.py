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
import atexit
import csv
import unittest
import os
from typing import Iterable, Union, TypeVar

import prettytable
from prettytable import PrettyTable

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
        filter(lambda x: x.event_type == 1 and x.on_gpu, prof.key_averages())
    )
    if len(gpu_kernel_items) == 0:
        return "-"
    kernel_gpu_time = sum(map(lambda x: x.gpu_time_total, gpu_kernel_items)) / prof.num
    return round(kernel_gpu_time, 1)


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


def print_summary_from_csv() -> None:
    print("----------------------------------------------------------------------")
    print('Summary ("KT" means "Kernel Time", "ET" means "End-to-end Time"):')
    with open(csv_filename, "r") as f:
        table: PrettyTable = prettytable.from_csv(f)
        table.field_names = [
            "OP",
            "Args",
            "Lib",
            "KT(GPU)",
            "KT(1 CPU)",
            "ET(1 CPU)",
            "KT(32 CPU)",
            "ET(32 CPU)",
            "Desc",
        ]
        for row in table.rows:
            row[2] = {"PyTorch": "PT", "OneFlow": "OF"}[row[2]]
        table.del_column("Desc")
        print(table)


# all functions registered are called in last in, first out order
atexit.register(print_summary_from_csv)

csv_filename = os.getenv("ONEFLOW_PROFILE_CSV", "op_prof")

if csv_filename[:-4] != ".csv":
    csv_filename += ".csv"

f = open(csv_filename, "w")
atexit.register(lambda f: f.close(), f)
writer = csv.writer(f)
writer.writerow(
    [
        "OP",
        "Args",
        "Library",
        "Kernel Time (us, GPU)",
        "Kernel Time (us, 1 CPU)",
        "End-to-end Time (us, 1 CPU)",
        "Kernel Time (us, 32 CPUs)",
        "End-to-end Time (us, 32 CPUs)",
        "Description",
    ]
)


auto_profiler.set_hardware_info_list([("cuda", None), ("cpu", 1), ("cpu", 32)])


def add_row(profs):
    op_name = get_sole_value([prof.op_name for prof in profs])
    args_description = get_sole_value([prof.args_description for prof in profs])
    additional_description = get_sole_value(
        [prof.additional_description for prof in profs]
    )
    writer.writerow(
        [
            op_name,
            args_description,
            "OneFlow",
            get_oneflow_gpu_kernel_time(profs[0]),
            get_oneflow_cpu_kernel_time(profs[1]),
            get_oneflow_cpu_end_to_end_time(profs[1]),
            get_oneflow_cpu_kernel_time(profs[2]),
            get_oneflow_cpu_end_to_end_time(profs[2]),
            additional_description,
        ]
    )
    writer.writerow(
        [
            op_name,
            args_description,
            "PyTorch",
            get_pytorch_gpu_kernel_time(profs[3]),
            get_pytorch_cpu_kernel_time(profs[4]),
            get_pytorch_cpu_end_to_end_time(profs[4]),
            get_pytorch_cpu_kernel_time(profs[5]),
            get_pytorch_cpu_end_to_end_time(profs[5]),
            additional_description,
        ]
    )
    f.flush()


auto_profiler.set_profiler_hook(add_row)

# Align with https://github.com/python/cpython/blob/3.10/Lib/unittest/__main__.py
__unittest = True

from unittest.main import main

loader = unittest.TestLoader()
loader.testMethodPrefix = "profile_"

main(module=None, testLoader=loader)
