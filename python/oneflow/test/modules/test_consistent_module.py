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

import numpy as np
import oneflow as flow
import oneflow.unittest

from test_module import np_relu
from oneflow.test_utils.automated_test_util import *


@autotest(n=1, check_graph=False)
def test_nested_module(test_case, placement, sbp):
    class CustomModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            return self.relu(x)

    m = CustomModule()
    m.train(random())
    x = random_tensor(ndim=2, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = m(x)
    return y


@autotest(n=1, check_graph=False)
def _test_relu(test_case, placement, sbp):
    relu = torch.nn.ReLU()
    x = random_tensor(ndim=2, dim0=8, dim1=8).to_consistent(placement, sbp)
    y = relu(x)
    return y


@autotest(n=1, auto_backward=False, check_graph=False)
def _test_load_state_dict(test_case, placement, sbp):
    class CustomModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = flow.nn.Parameter(flow.Tensor(8, 8).to_consistent(placement, sbp))

        def forward(self, x):
            return self.w

    m = CustomModule()
    ones = flow.ones(8, 8).to_consistent(placement, sbp).numpy()
    m.load_state_dict({"w": ones})
    x = flow.Tensor(2, 3).to_consistent(placement, sbp)
    y = m(x).numpy()
    test_case.assertTrue(np.array_equal(y, ones))


def _test_parameter(test_case, placement, sbp):
    t = flow.Tensor(8, 8).to_consistent(placement, sbp)
    p = flow.nn.Parameter(t)
    test_case.assertEqual(type(p), flow.nn.Parameter)
    test_case.assertEqual(p.shape, t.shape)


@autotest(n=1, auto_backward=False)
def _test_module_setattr(test_case, placement, sbp):
    class CustomModule(flow.nn.Module):
        def __init__(self, param1, param2):
            super().__init__()
            self.param1 = param1
            self.param2 = param2

    param0 = flow.nn.Parameter(flow.Tensor(8, 8).to_consistent(placement, sbp))
    param1 = flow.nn.Parameter(flow.Tensor(8, 8).to_consistent(placement, sbp))
    param2 = CustomModule(param0, param1)
    m = CustomModule(param1, param2)
    params = list(m.parameters())
    test_case.assertEqual(len(params), 2)

    test_case.assertTrue(
        np.allclose(params[0].numpy(), param1.numpy(), atol=1e-4, rtol=1e-4)
    )
    test_case.assertTrue(
        np.allclose(params[1].numpy(), param0.numpy(), atol=1e-4, rtol=1e-4)
    )
    children = list(m.children())
    test_case.assertEqual(len(children), 1)
    child = children[0]
    test_case.assertEqual(child, param2)
    child_params = list(child.parameters())

    test_case.assertEqual(len(child_params), 2)
    print("child 0 param = ", child_params[0].numpy())
    print("param0 = ", param0.numpy())
    test_case.assertTrue(np.allclose(child_params[0].numpy(), param0.numpy()))
    test_case.assertTrue(np.allclose(child_params[1].numpy(), param1.numpy()))


def _test_module_float_double(test_case, placement, sbp):
    class CustomModule(flow.nn.Module):
        def __init__(self, param1, param2):
            super().__init__()
            self.param1 = param1
            self.param2 = param2

    tensor0 = flow.nn.Parameter(
        flow.Tensor(8, 8).to(dtype=flow.float64).to_consistent(placement, sbp)
    )
    tensor1 = flow.nn.Parameter(
        flow.Tensor(8, 8).to(dtype=flow.float64).to_consistent(placement, sbp)
    )
    m = CustomModule(tensor0, tensor1).to_consistent(placement, sbp)
    m = m.float()
    state_dict = m.state_dict()
    test_case.assertEqual(state_dict["param1"].dtype, flow.float32)
    test_case.assertEqual(state_dict["param2"].dtype, flow.float32)

    m = m.double()
    state_dict = m.state_dict()
    test_case.assertEqual(state_dict["param1"].dtype, flow.float64)
    test_case.assertEqual(state_dict["param2"].dtype, flow.float64)


class TestModule(flow.unittest.TestCase):
    @consistent
    def test_module(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                # TODO(): unknown error
                test_nested_module(test_case, placement, sbp)
                _test_relu(test_case, placement, sbp)
                _test_load_state_dict(test_case, placement, sbp)
                _test_parameter(test_case, placement, sbp)
                _test_save_state_dict(test_case, placement, sbp)
                # TODO(): unknown error
                # _test_module_setattr(test_case, placement, sbp)
                _test_module_float_double(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
