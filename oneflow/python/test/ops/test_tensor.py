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
import oneflow as flow
import numpy as np
import os
import random
import oneflow.typing as oft
from collections import OrderedDict


def fake_flow_ones(shape):
    tensor = flow.Tensor(*shape)
    tensor.set_data_initializer(flow.ones_initializer())
    return tensor


@flow.unittest.skip_unless_1n1d()
class TestTensor(flow.unittest.TestCase):
    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_numpy(test_case):
        shape = (2, 3)
        test_case.assertTrue(
            np.array_equal(
                fake_flow_ones(shape).numpy(), np.ones(shape, dtype=np.float32)
            )
        )

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_init(test_case):
        shape = (2, 3)
        x = flow.Tensor(*shape)

        x.fill_(5)
        test_case.assertTrue(np.array_equal(x.numpy(), 5 * np.ones(x.shape)))

        flow.nn.init.ones_(x)
        test_case.assertTrue(np.array_equal(x.numpy(), np.ones(x.shape)))

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_creating_consistent_tensor(test_case):
        shape = (2, 3)
        x = flow.Tensor(*shape, placement=flow.placement("gpu", ["0:0"], None))
        x.set_placement(flow.placement("cpu", ["0:0"], None))
        x.set_is_consistent(True)
        test_case.assertTrue(not x.is_cuda)
        x.determine()

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_user_defined_data(test_case):
        list_data = [5, 5]
        tuple_data = (5, 5)
        numpy_data = np.array((5, 5))
        x = flow.Tensor(list_data)
        y = flow.Tensor(tuple_data)
        z = flow.Tensor(numpy_data)

        test_case.assertTrue(np.array_equal(x.numpy(), 5 * np.ones(x.shape)))
        test_case.assertTrue(np.array_equal(y.numpy(), 5 * np.ones(x.shape)))
        test_case.assertTrue(np.array_equal(z.numpy(), 5 * np.ones(x.shape)))


if __name__ == "__main__":
    unittest.main()
