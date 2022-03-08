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


def _test_nested_module(test_case, placement, sbp):
    class CustomModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = flow.nn.ReLU()

        def forward(self, x):
            return self.relu(x)

    m = CustomModule().to_global(placement, sbp)
    m.train(random())
    x = random_tensor(ndim=2, dim0=8, dim1=8).to_global(placement, sbp).oneflow
    y = m(x)
    test_case.assertTrue(np.array_equal(np_relu(x.numpy()), y.numpy()))


@autotest(n=1, check_graph=False)
def _test_relu(test_case, placement, sbp):
    relu = torch.nn.ReLU()
    x = random_tensor(ndim=2, dim0=8, dim1=8).to_global(placement, sbp)
    y = relu(x)
    return y


def _test_load_state_dict(test_case, placement, sbp):
    class CustomModule(flow.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = flow.nn.Parameter(
                random_tensor(ndim=2, dim0=8, dim1=8).to_global(placement, sbp).oneflow
            )

        def forward(self):
            return self.w

    m = CustomModule().to_global(placement, sbp)
    ones = (
        random_tensor(ndim=2, dim0=8, dim1=8).to_global(placement, sbp).oneflow.numpy()
    )
    m.load_state_dict({"w": ones})
    y = m().numpy()
    test_case.assertTrue(np.array_equal(y, ones))


def _test_parameter(test_case, placement):
    ndim = random().to(int).value()
    dim_list = [random().to(int).value() * 8 for _ in range(ndim)]
    sbp = random_sbp(placement, max_dim=ndim)
    t = random_tensor(ndim, *dim_list).to_global(placement, sbp).oneflow
    p = flow.nn.Parameter(t)
    test_case.assertEqual(type(p), flow.nn.Parameter)
    test_case.assertEqual(p.shape, t.shape)


def _test_module_float_double(test_case, placement, sbp):
    class CustomModule(flow.nn.Module):
        def __init__(self, param1, param2):
            super().__init__()
            self.param1 = param1
            self.param2 = param2

    tensor0 = flow.nn.Parameter(flow.Tensor(8, 8).to(dtype=flow.float64))
    tensor1 = flow.nn.Parameter(flow.Tensor(8, 8).to(dtype=flow.float64))
    m = CustomModule(tensor0, tensor1).to_global(placement, sbp)
    m = m.float()
    state_dict = m.state_dict()
    test_case.assertEqual(state_dict["param1"].dtype, flow.float32)
    test_case.assertEqual(state_dict["param2"].dtype, flow.float32)

    m = m.double()
    state_dict = m.state_dict()
    test_case.assertEqual(state_dict["param1"].dtype, flow.float64)
    test_case.assertEqual(state_dict["param2"].dtype, flow.float64)


def _test_module_placement_change(test_case, placement, sbp):
    class CustomModule(flow.nn.Module):
        def __init__(self, param1, param2):
            super().__init__()
            self.param1 = param1
            self.param2 = param2  

    tensor0 = flow.nn.Parameter(flow.Tensor(2, 3, device=flow.device("cpu")))
    tensor1 = flow.nn.Parameter(flow.Tensor(2, 3, device=flow.device("cpu")))
    m = CustomModule(tensor0, tensor1).to_global(placement, sbp)
    state_dict = m.state_dict()
    test_case.assertEqual(state_dict["param1"].placement, placement)
    test_case.assertEqual(state_dict["param2"].placement, placement)


class TestModule(flow.unittest.TestCase):
    @globaltest
    def test_module(test_case):
        for placement in all_placement():
            _test_parameter(test_case, placement)
            for sbp in all_sbp(placement, max_dim=2):
                # test_module.py test cpu only.
                if placement.type == "cpu":
                    _test_nested_module(test_case, placement, sbp)
                _test_relu(test_case, placement, sbp)
                _test_load_state_dict(test_case, placement, sbp)
                _test_module_float_double(test_case, placement, sbp)
                _test_module_placement_change(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
