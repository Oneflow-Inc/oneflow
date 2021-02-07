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
from typing import Tuple

import numpy as np

import oneflow as flow
import oneflow.typing as tp


_counter = 0
def get_var_helper():
    global _counter
    var = flow.get_variable("x_" + str(_counter), shape=(2, 3), initializer=flow.kaiming_initializer((2, 3)))
    _counter += 1
    return var


class TestModule(flow.unittest.TestCase):
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

    def test_forward_with_variable(test_case):
        class AddTo(flow.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w

            def forward(self, x):
                return x + self.w

        @flow.global_function()
        def job() -> Tuple[tp.Numpy, tp.Numpy]:
            w = get_var_helper()
            x = get_var_helper()
            m = AddTo(w)
            return m(x), w + x

        res1, res2 = job()
        test_case.assertTrue(np.array_equal(res1, res2))

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

        param0 = flow.nn.Parameter()
        param1 = flow.nn.Parameter()
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

    def test_state_dict(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self, param1, param2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

        param0 = flow.nn.Parameter()
        param1 = flow.nn.Parameter()
        param2 = CustomModule(param0, param1)
        m = CustomModule(param1, param2)

        state_dict = m.state_dict()
        print(state_dict)
        test_case.assertEqual(len(state_dict), 3)


    # TODO: add more tests about module api


if __name__ == "__main__":
    unittest.main()
