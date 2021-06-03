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


def _test_ones(test_case, device, shape):
    y = flow.ones(shape)
    test_case.assertTrue(np.array_equal(np.ones(shape), y.numpy()))

    y2 = flow.ones(10)
    test_case.assertTrue(np.array_equal(np.ones(10), y2.numpy()))

    y3 = flow.ones(10, dtype=flow.float64)
    test_case.assertTrue(np.array_equal(np.ones(10, dtype=np.float64), y3.numpy()))


def _test_zeros(test_case, device, shape):
    y = flow.zeros(shape)
    test_case.assertTrue(np.array_equal(np.zeros(shape), y.numpy()))

    y2 = flow.zeros(10)
    test_case.assertTrue(np.array_equal(np.zeros(10), y2.numpy()))

    y3 = flow.zeros(10, dtype=flow.int)
    test_case.assertTrue(np.array_equal(np.zeros(10, dtype=int), y3.numpy()))


def _test_ones_like(test_case, device, shape):
    x = flow.Tensor(np.ones(shape, dtype=np.float64))
    test_case.assertTrue(
        np.array_equal(np.ones_like(x.numpy()), flow.ones_like(x).numpy())
    )

    x2 = flow.Tensor(np.ones([2, 4], dtype=int))
    test_case.assertTrue(
        np.array_equal(np.ones_like(x2.numpy()), flow.ones_like(x2).numpy())
    )


def _test_zeros_like(test_case, device, shape):
    x = flow.Tensor(np.ones(shape, dtype=np.float64))
    test_case.assertTrue(
        np.array_equal(np.zeros_like(x.numpy()), flow.zeros_like(x).numpy())
    )

    x2 = flow.Tensor(np.ones(shape, dtype=int))
    test_case.assertTrue(
        np.array_equal(np.zeros_like(x2.numpy()), flow.zeros_like(x2).numpy())
    )


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestConstantModule(flow.unittest.TestCase):
    def test_cast(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_ones,
            _test_zeros,
            _test_ones_like,
            _test_zeros_like,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
