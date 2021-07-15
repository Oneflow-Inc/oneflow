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
from automated_test_util import *
import torch


def arccosh_input_tensor(shape):
    def generator(_):
        low = 1
        high = 2
        rng = np.random.default_rng()
        np_arr = rng.random(size=shape) * (high - low) + low
        return (
            flow.Tensor(np_arr, dtype=flow.float32),
            torch.tensor(np_arr, dtype=torch.float32),
        )

    return generator


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestArccosh(flow.unittest.TestCase):
    def test_arccosh_flow_with_random_data(test_case):
        for device in ["cpu", "cuda"]:
            test_flow_against_pytorch(
                test_case,
                "arccosh",
                device=device,
                n=2,
                extra_generators={"input": arccosh_input_tensor((3, 3))},
            )

    def test_acosh_tensor_with_random_data(test_case):
        for device in ["cpu", "cuda"]:
            test_tensor_against_pytorch(
                test_case,
                "arccosh",
                device=device,
                n=2,
                extra_generators={"input": arccosh_input_tensor((3, 3))},
            )


if __name__ == "__main__":
    unittest.main()
