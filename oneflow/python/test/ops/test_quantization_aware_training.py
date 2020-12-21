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
import unittest
from collections import OrderedDict

import numpy as np

import oneflow as flow
import oneflow.typing as oft
from test_util import GenArgList


def _test(test_case, per_channel, symmetric, build_backbone_fn):
    flow.clear_default_session()

    flow.config.enable_debug_mode(True)
    func_config = flow.FunctionConfig()
    func_config.enable_qat(True)
    func_config.qat.symmetric(symmetric)
    func_config.qat.per_channel_weight_quantization(per_channel)
    func_config.qat.moving_min_max_stop_update_after_iters(1000)

    INPUT_SHAPE = (2, 8, 10, 12)

    @flow.global_function(type="train", function_config=func_config)
    def Foo(x: oft.Numpy.Placeholder(INPUT_SHAPE)):
        y = build_backbone_fn(x)
        flow.optimizer.SGD(
            flow.optimizer.PiecewiseConstantScheduler([], [5]), momentum=0
        ).minimize(y)

    Foo(np.ones(INPUT_SHAPE, dtype=np.float32))


class TestQAT(flow.unittest.TestCase):
    def test_qat(test_case):
        def build_conv_with_bias(x):
            return flow.layers.conv2d(x, 4, 3, 1, "SAME", use_bias=True)

        def build_conv_without_bias(x):
            return flow.layers.conv2d(x, 4, 3, 1, "SAME", use_bias=False)

        arg_dict = OrderedDict()
        arg_dict["per_channel"] = [True, False]
        arg_dict["symmetric"] = [True, False]
        arg_dict["build_backbone_fn"] = [build_conv_with_bias, build_conv_without_bias]
        for arg in GenArgList(arg_dict):
            _test(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
