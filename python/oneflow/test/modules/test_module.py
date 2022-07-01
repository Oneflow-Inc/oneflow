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

import os
import warnings
import tempfile
import unittest
from itertools import repeat
from typing import Tuple, Union, List
from collections import OrderedDict

import numpy as np

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest


def np_relu(np_arr):
    return np.where(np_arr > 0, np_arr, 0)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestModule(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
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

    @flow.unittest.skip_unless_1n1d()
    def test_relu(test_case):
        relu = flow.nn.ReLU()
        x = flow.Tensor(2, 3)
        flow.nn.init.uniform_(x, a=-1.0, b=1.0)
        y = relu(x)
        test_case.assertTrue(np.array_equal(np_relu(x.numpy()), y.numpy()))

    @flow.unittest.skip_unless_1n1d()
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

    @flow.unittest.skip_unless_1n1d()
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

    @flow.unittest.skip_unless_1n1d()
    def test_parameter(test_case):
        shape = (3, 4)
        t = flow.Tensor(*shape)
        p = flow.nn.Parameter(t)
        test_case.assertEqual(type(p), flow.nn.Parameter)
        test_case.assertEqual(p.shape, shape)

    @flow.unittest.skip_unless_1n1d()
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

    @flow.unittest.skip_unless_1n1d()
    def test_train_eval(test_case):
        m = flow.nn.Module()
        test_case.assertEqual(m.training, True)
        m.train()
        test_case.assertEqual(m.training, True)
        m.eval()
        test_case.assertEqual(m.training, False)

    @flow.unittest.skip_unless_1n1d()
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
        test_case.assertTrue(np.allclose(child_params[0].numpy(), param0.numpy()))
        test_case.assertTrue(np.allclose(child_params[1].numpy(), param1.numpy()))

    @flow.unittest.skip_unless_1n1d()
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

    @flow.unittest.skip_unless_1n1d()
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
            # Creates a new file and test fault tolerance
            with open(os.path.join(save_dir, "random_file"), "w") as fp:
                fp.write("nothing")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded_state_dict = flow.load(save_dir)
            m.load_state_dict(loaded_state_dict)
        res2 = m()
        test_case.assertTrue(np.array_equal(res1.numpy(), res2.numpy()))

    def _test_save_and_load_global_from_nested_dict(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = flow.nn.Parameter(flow.randn(3, 32, 3, 3))

            def forward(self):
                return self.param

        m1 = CustomModule()
        m1 = m1.to_global(
            flow.placement("cuda", range(1, 3)), flow.sbp.broadcast
        ).to_global(sbp=flow.sbp.split(1))
        m2 = CustomModule()
        m2 = m2.to_global(flow.placement("cuda", range(1, 3)), flow.sbp.broadcast)
        res1 = m1() + m2()
        state_dict1 = m1.state_dict()
        state_dict2 = m2.state_dict()
        state_dict = {"m1": state_dict1, "m2": state_dict2}

        with tempfile.TemporaryDirectory() as f:
            with test_case.assertRaises(Exception):
                flow.save(state_dict, f)

            global_src_dst_rank = 0
            flow.save(state_dict, f, global_dst_rank=global_src_dst_rank)
            rank = flow.env.get_rank()
            if rank != global_src_dst_rank:
                test_case.assertEqual(len(os.listdir(f)), 0)

            m1 = CustomModule()
            m1 = m1.to_global(
                flow.placement("cuda", [[0, 1], [2, 3]]),
                [flow.sbp.broadcast, flow.sbp.broadcast],
            ).to_global(sbp=[flow.sbp.split(1), flow.sbp.broadcast])
            m2 = CustomModule()
            m2 = m2.to_global(
                flow.placement("cuda", [[0, 1], [2, 3]]),
                [flow.sbp.broadcast, flow.sbp.broadcast],
            ).to_global(sbp=[flow.sbp.broadcast, flow.sbp.split(1)])

            with test_case.assertRaises(Exception):
                loaded_state_dict = flow.load(f)
                m1.load_state_dict(loaded_state_dict["m1"])

            loaded_state_dict = flow.load(f, global_src_rank=global_src_dst_rank)
            test_case.assertEqual(len(loaded_state_dict), 2)
            m1.load_state_dict(loaded_state_dict["m1"])
            m2.load_state_dict(loaded_state_dict["m2"])
            res2 = m1() + m2()

        test_case.assertTrue(np.array_equal(res1.numpy(), res2.numpy()))

    @flow.unittest.skip_unless_1n4d()
    def test_save_and_load_global_from_nested_dict_1n4d(test_case):
        test_case._test_save_and_load_global_from_nested_dict()

    @flow.unittest.skip_unless_2n2d()
    def test_save_and_load_global_from_nested_dict_2n2d(test_case):
        test_case._test_save_and_load_global_from_nested_dict()

    @flow.unittest.skip_unless_1n1d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_module_cpu_cuda(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self, param1, param2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

        tensor0 = flow.nn.Parameter(flow.Tensor(2, 3, device=flow.device("cpu")))
        tensor1 = flow.nn.Parameter(flow.Tensor(2, 3, device=flow.device("cpu")))
        sub_module = CustomModule(tensor0, tensor1)
        m = CustomModule(tensor1, sub_module)
        m.cuda()
        state_dict = m.state_dict()
        test_case.assertEqual(state_dict["param2.param1"].device, flow.device("cuda:0"))
        test_case.assertEqual(state_dict["param2.param2"].device, flow.device("cuda:0"))

        m.cpu()
        state_dict = m.state_dict()
        test_case.assertEqual(state_dict["param2.param1"].device, flow.device("cpu"))
        test_case.assertEqual(state_dict["param2.param2"].device, flow.device("cpu"))

    @flow.unittest.skip_unless_1n1d()
    def test_module_float_double(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self, param1, param2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

        tensor0 = flow.nn.Parameter(flow.Tensor(2, 3).to(dtype=flow.float64))
        tensor1 = flow.nn.Parameter(flow.Tensor(2, 3).to(dtype=flow.float64))
        m = CustomModule(tensor0, tensor1)
        m = m.float()
        state_dict = m.state_dict()
        test_case.assertEqual(state_dict["param1"].dtype, flow.float32)
        test_case.assertEqual(state_dict["param2"].dtype, flow.float32)

        m = m.double()
        state_dict = m.state_dict()
        test_case.assertEqual(state_dict["param1"].dtype, flow.float64)
        test_case.assertEqual(state_dict["param2"].dtype, flow.float64)

    @flow.unittest.skip_unless_1n1d()
    def test_moduledict(test_case):
        class ModuleDict(nn.Module):
            def __init__(self):
                super(ModuleDict, self).__init__()
                self.choices = nn.ModuleDict(
                    {"conv": nn.Conv2d(10, 10, 3), "pool": nn.MaxPool2d(3)}
                )
                self.activations = nn.ModuleDict(
                    {"relu": nn.ReLU(), "prelu": nn.PReLU()}
                )

            def forward(self, x, choice, act):
                x = self.choices[choice](x)
                x = self.activations[act](x)
                return x

        model = ModuleDict()
        input = flow.tensor(np.random.randn(4, 10, 32, 32), dtype=flow.float32)
        output = model(input, "conv", "relu")
        test_case.assertEqual(output.shape, flow.Size([4, 10, 30, 30]))


if __name__ == "__main__":
    unittest.main()
