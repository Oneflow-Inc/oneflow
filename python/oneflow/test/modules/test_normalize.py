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
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList, type_name_to_flow_type
from oneflow.test_utils.automated_test_util import *


def _test_functional_normalize_double_dtype(test_case, device, dtype):
    dtype = type_name_to_flow_type[dtype]
    x = flow.ones(2, 2, dtype=dtype).to(device)
    y = flow.nn.functional.normalize(x, p=2, dim=0)
    test_case.assertEqual((2, 2), y.shape)
    out = np.array(
        [
            [0.7071067690849304, 0.7071067690849304],
            [0.7071067690849304, 0.7071067690849304],
        ]
    )
    test_case.assertTrue(np.allclose(y.numpy().tolist(), out, 1e-05, 1e-05))


@flow.unittest.skip_unless_1n1d()
class TestFunctionalNormalize(flow.unittest.TestCase):
    def test_functional_normalize_naive(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [_test_functional_normalize_double_dtype]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = ["float32", "double"]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5)
    def test_functional_normalize(test_case):
        device = random_device()
        ndim = random(low=2)

        shape = list(random_tensor(ndim=ndim).oneflow.shape)
        dim = random(low=0, high=ndim).to(int).value()
        shape[dim] = random(low=2, high=8).to(int).value()
        shape = tuple(shape)

        x = random_tensor(len(shape), *shape).to(device)
        y = torch.nn.functional.normalize(x, oneof(2, 3, 4), dim, 1e-12)

        return y


if __name__ == "__main__":
    unittest.main()
