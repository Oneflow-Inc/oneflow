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
import sys
import subprocess
import tempfile

import oneflow as flow
import oneflow.test_utils.automated_test_util.profiler as auto_profiler
from oneflow.autoprof.util import *


csv_filename = os.getenv("ONEFLOW_PROFILE_CSV", "op_prof")

if csv_filename[-4:] != ".csv":
    csv_filename += ".csv"

f = open(csv_filename, "w")
# all functions registered are called in last in, first out order
if flow.support.env_var_util.parse_boolean_from_env(
    "ONEFLOW_PROFILE_PRINT_SUMMARY", True
):
    atexit.register(print_summary_from_csv, csv_filename)
atexit.register(lambda f: f.close(), f)

writer = csv.writer(f)

ONLY_ONEFLOW = flow.support.env_var_util.parse_boolean_from_env(
    "ONEFLOW_PROFILE_ONLY_ONEFLOW", False
)
ONLY_PYTORCH = flow.support.env_var_util.parse_boolean_from_env(
    "ONEFLOW_PROFILE_ONLY_PYTORCH", False
)
assert not (ONLY_ONEFLOW and ONLY_PYTORCH)

if not ONLY_ONEFLOW and not ONLY_PYTORCH:
    env = os.environ.copy()
    env.update({"ONEFLOW_PROFILE_ONLY_ONEFLOW": "1"})
    temp_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    env.update({"ONEFLOW_PROFILE_CSV": temp_f.name})
    env.update({"ONEFLOW_PROFILE_PRINT_SUMMARY": "0"})
    subprocess.run([sys.executable, "-m", "oneflow.autoprof", *sys.argv[1:]], env=env)
    temp_f.close()
    temp_f = open(temp_f.name, "r")
    rows = list(csv.reader(temp_f))
    temp_f.close()
    os.remove(temp_f.name)

    env = os.environ.copy()
    env.update({"ONEFLOW_PROFILE_ONLY_PYTORCH": "1"})
    temp_f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    env.update({"ONEFLOW_PROFILE_CSV": temp_f.name})
    env.update({"ONEFLOW_PROFILE_PRINT_SUMMARY": "0"})
    subprocess.run([sys.executable, "-m", "oneflow.autoprof", *sys.argv[1:]], env=env)
    temp_f.close()
    temp_f = open(temp_f.name, "r")
    rows.extend(list(csv.reader(temp_f))[1:])
    temp_f.close()
    os.remove(temp_f.name)

    writer.writerows(rows)
    exit(0)

writer.writerow(
    [
        "OP",
        "Args",
        "Library",
        "Kernel Time (us, GPU)",
        "Kernel Bandwidth (GB/s, GPU)",
        "Kernel Time (us, 1 CPU)",
        "End-to-end Time (us, 1 CPU)",
        "Kernel Time (us, 32 CPUs)",
        "End-to-end Time (us, 32 CPUs)",
        "Description",
    ]
)

auto_profiler.set_hardware_info_list([("cuda", None), ("cpu", 1), ("cpu", 32)])

if ONLY_ONEFLOW:
    auto_profiler.profiled_framework = ["oneflow"]
if ONLY_PYTORCH:
    auto_profiler.profiled_framework = ["pytorch"]

auto_profiler.set_profiler_hook(lambda profs: add_row(profs, writer, f))

# Align with https://github.com/python/cpython/blob/3.10/Lib/unittest/__main__.py
__unittest = True

from unittest.main import main

loader = unittest.TestLoader()
loader.testMethodPrefix = "profile_"

main(module=None, testLoader=loader)
