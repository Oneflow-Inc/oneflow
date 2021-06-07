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

import oneflow.experimental as flow
from test_util import GenArgList


def np_mseloss(np_input, np_target):
    np_mse = np.square(np_target - np_input)
    np_mse_mean = np.mean(np_mse)
    np_mse_sum = np.sum(np_mse)

    return {
        "none": np_mse,
        "mean": np_mse_mean,
        "sum": np_mse_sum,
    }


def _test_mseloss(test_case, device, reduction):
    x = np.random.randn(2, 3)
    y = np.random.randn(2, 3)
    input = flow.Tensor(x, dtype=flow.float32, device=flow.device(device))
    target = flow.Tensor(y, dtype=flow.float32, device=flow.device(device))

    loss = flow.nn.MSELoss(reduction=reduction)
    loss = loss.to(device)
    of_out = loss(input, target)
    np_out = np_mseloss(x, y)[reduction]
    test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-4, 1e-4))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestMSELossModule(flow.unittest.TestCase):
    def test_mseloss(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_mseloss,
        ]
        arg_dict["device"] = ["cpu"]
        arg_dict["reduction"] = ["none"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
