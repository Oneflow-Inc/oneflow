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
import tensorflow as tf
import test_global_storage
from test_util import Args, GenArgDict, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as oft
import unittest


def test_train_consistent(test_case):
    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.default_logical_view(flow.scope.consistent_view())
    func_config.default_data_type(flow.float32)

    @flow.global_function(type="train", function_config=func_config)
    def Foo(
        x: oft.Numpy.Placeholder((2, 8, 32, 32)),
        add_in: oft.Numpy.Placeholder((2, 8, 32, 32)),
    ):
        y = flow.layers.batch_normalization_add_relu(x, add_in, axis=1)
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [0.001]), momentum=0
        ).minimize(flow.math.reduce_sum(y))

    Foo(
        np.ones((2, 8, 32, 32), dtype=np.float32),
        np.ones((2, 8, 32, 32), dtype=np.float32),
    )
