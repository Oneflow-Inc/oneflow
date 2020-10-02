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
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow


class MemoryZoneOutOfMemoryException(Exception):
    def __init__(self, err="memory_zone_out_of_memory"):
        Exception.__init__(self, err)


def constant(device_type):
    flow.env.init()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()

    @flow.global_function(function_config=func_config)
    def ConstantJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.constant(
                6, dtype=flow.float, shape=(1024 * 1024 * 1024, 1024 * 1024 * 1024)
            )
            return x

    try:
        ConstantJob().get()
    except Exception as e:
        if "memory_zone_out_of_memory" in str(e):
            print(e)
            raise MemoryZoneOutOfMemoryException()


# def test_MemoryZoneOutOfMemoryOfCpu(test_case):
# test_case.assertRaises(MemoryZoneOutOfMemoryException, memory_zone_out_of_memory_of_cpu)


# def test_MemoryZoneOutOfMemoryOfGpu(test_case):
# test_case.assertRaises(MemoryZoneOutOfMemoryException, memory_zone_out_of_memory_of_gpu)


def memory_zone_out_of_memory_of_gpu():
    return constant("gpu")


def memory_zone_out_of_memory_of_cpu():
    return constant("cpu")
