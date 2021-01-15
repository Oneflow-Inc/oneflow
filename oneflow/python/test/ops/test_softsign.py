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
import unittest
import oneflow as flow
import numpy as np
import oneflow.typing as tp
from test_util import GenArgList
from collections import OrderedDict
from typing import Dict


def compare_softsign_with_np(
    input_shape, device_type, value_type, machine_ids, device_counts
):
    if value_type[1] == flow.float16:
        input_1 = np.random.uniform(-1, 7, size=input_shape).astype(np.float16)
        input_1 = np.array(input_1, dtype=value_type[0])
    else:
        input_1 = np.random.uniform(-1, 7, size=input_shape).astype(value_type[0])
    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()

    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)
    
    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))

    if value_type == flow.float16:
        func_config.default_data_type(flow.float32)
    else:
        func_config.default_data_type(value_type[1])
    
    