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
import collections.abc
from itertools import repeat
import unittest
from typing import Tuple, Union
import tempfile

import numpy as np

import oneflow.experimental as flow
import oneflow.typing as tp


def np_relu(np_arr):
    return np.where(np_arr > 0, np_arr, 0)


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    def test_nested_module(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = flow.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        m = CustomModule()
        x = flow.Tensor(2, 3)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        y = m(x)
        test_case.assertTrue(np.array_equal(np_relu(x.numpy()), y.numpy()))

    def test_relu(test_case):
        relu = flow.nn.ReLU()

        x = flow.Tensor(2, 3)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        y = relu(x)
        test_case.assertTrue(np.array_equal(np_relu(x.numpy()), y.numpy()))

    def test_load_state_dict(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = flow.nn.Parameter(flow.Tensor(2, 3))

            def forward(self, x):
                return self.w

        m = CustomModule()

        ones = np.ones((2, 3), dtype=np.float32)
        m.load_state_dict({"w": ones})
        x = flow.Tensor(2, 3)
        y = m(x).numpy()

        test_case.assertTrue(np.array_equal(y, ones))

    def test_state_dict(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self, param1, param2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

        tensor0 = flow.nn.Parameter(flow.Tensor(2, 3))
        tensor1 = flow.nn.Parameter(flow.Tensor(2, 3))
        sub_module = CustomModule(tensor0, tensor1)
        m = CustomModule(tensor1, sub_module)

        state_dict = m.state_dict()
        test_case.assertEqual(
            state_dict,
            {"param2.param1": tensor0, "param2.param2": tensor1, "param1": tensor1},
        )

    def test_parameter(test_case):
        shape = (3, 4)
        t = flow.Tensor(*shape)
        p = flow.nn.Parameter(t)
        test_case.assertEqual(type(p), flow.nn.Parameter)
        test_case.assertEqual(p.shape, shape)

    def test_module_forward(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w

            def forward(self, x):
                return x + self.w

        m = CustomModule(5)
        test_case.assertEqual(m(1), 6)

        m = CustomModule(4)
        test_case.assertEqual(m(3), 7)

    def test_train_eval(test_case):
        m = flow.nn.Module()
        test_case.assertEqual(m.training, True)
        m.train()
        test_case.assertEqual(m.training, True)
        m.eval()
        test_case.assertEqual(m.training, False)

    def test_module_setattr(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self, param1, param2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

        param0 = flow.nn.Parameter(flow.Tensor(2, 3))
        param1 = flow.nn.Parameter(flow.Tensor(2, 3))
        param2 = CustomModule(param0, param1)
        m = CustomModule(param1, param2)

        # m.parameters() contains param0 + param1 in submodule param2
        # and param1 in m
        params = list(m.parameters())
        test_case.assertEqual(len(params), 2)
        test_case.assertEqual(params[0], param1)
        test_case.assertEqual(params[1], param0)

        children = list(m.children())
        test_case.assertEqual(len(children), 1)
        child = children[0]
        test_case.assertEqual(child, param2)

        child_params = list(child.parameters())
        test_case.assertEqual(len(child_params), 2)
        test_case.assertEqual(child_params[0], param0)
        test_case.assertEqual(child_params[1], param1)

    def test_module_apply(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.modules = flow.nn.Module()

        global module_num
        module_num = 0

        def get_module_num(m):
            global module_num
            module_num += 1

        net = CustomModule()
        net.apply(get_module_num)

        test_case.assertEqual(module_num, 2)

    def test_save_state_dict(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.param1 = flow.nn.Parameter(flow.Tensor(32, 1024, 1024))
                self.param2 = flow.nn.Parameter(flow.Tensor(32, 1024, 1024))

            def forward(self):
                return self.param1 + self.param2

        m = CustomModule()

        res1 = m()
        state_dict = m.state_dict()
        with tempfile.TemporaryDirectory() as save_dir:
            flow.save(state_dict, save_dir)
            loaded_state_dict = flow.load(save_dir)
            m.load_state_dict(loaded_state_dict)
        res2 = m()
        test_case.assertTrue(np.array_equal(res1.numpy(), res2.numpy()))


if __name__ == "__main__":
    unittest.main()
