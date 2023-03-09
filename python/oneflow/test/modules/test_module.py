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
import math
import warnings
import tempfile
import unittest
from itertools import repeat
from typing import Tuple, Union, List
from collections import OrderedDict

import numpy as np
import torch

import oneflow as flow
import oneflow.nn as nn
import oneflow.unittest
from oneflow._oneflow_internal import TensorTuple
from oneflow.test_utils.test_util import GenArgList


def np_relu(np_arr):
    return np.where(np_arr > 0, np_arr, 0)


def _test_hooks(test_case, backward_register_fn):
    module = nn.Sigmoid()
    input = flow.ones(5, 5, requires_grad=True)

    counter = {"forwards": 0, "backwards": 0}

    def fw_hook(inc, h_module, input, output):
        test_case.assertTrue(isinstance(input, tuple))
        test_case.assertTrue(isinstance(output, flow.Tensor))
        test_case.assertTrue(h_module is module)
        test_case.assertTrue(flow.equal(input[0], flow.ones(5, 5)))
        test_case.assertTrue(
            flow.equal(output, flow.empty(5, 5).fill_(1 / (1 + 1 / math.e)))
        )
        counter["forwards"] += inc

    def bw_hook(inc, h_module, grad_input, grad_output):
        test_case.assertTrue(isinstance(grad_input, TensorTuple))
        test_case.assertTrue(isinstance(grad_output, TensorTuple))
        test_case.assertTrue(h_module is module)
        test_case.assertTrue(flow.equal(grad_output[0], flow.ones(5, 5) * 2))
        counter["backwards"] += inc

    test_fwd = module.register_forward_hook(lambda *args: fw_hook(1, *args))

    module(input)
    module(input)
    test_case.assertEqual(counter["forwards"], 2)
    test_case.assertEqual(counter["backwards"], 0)

    test_bwd = getattr(module, backward_register_fn)(lambda *args: bw_hook(1, *args))

    output = module(input)
    test_case.assertEqual(counter["forwards"], 3)
    test_case.assertEqual(counter["backwards"], 0)

    output.backward(flow.ones(5, 5) * 2, retain_graph=True)
    test_case.assertEqual(counter["forwards"], 3)
    test_case.assertEqual(counter["backwards"], 1)

    output.backward(flow.ones(5, 5) * 2, retain_graph=True)
    test_case.assertEqual(counter["forwards"], 3)
    test_case.assertEqual(counter["backwards"], 2)

    test2_fwd = module.register_forward_hook(lambda *args: fw_hook(2, *args))

    output = module(input)
    test_case.assertEqual(counter["forwards"], 6)
    test_case.assertEqual(counter["backwards"], 2)

    test2_bwd = getattr(module, backward_register_fn)(lambda *args: bw_hook(2, *args))
    module(input).backward(flow.ones(5, 5) * 2)
    test_case.assertEqual(counter["forwards"], 9)
    test_case.assertEqual(counter["backwards"], 5)

    test2_bwd.remove()

    module(input).backward(flow.ones(5, 5) * 2)
    test_case.assertEqual(counter["forwards"], 12)
    test_case.assertEqual(counter["backwards"], 6)

    test2_fwd.remove()

    module(input).backward(flow.ones(5, 5) * 2)
    test_case.assertEqual(counter["forwards"], 13)
    test_case.assertEqual(counter["backwards"], 7)

    test_fwd.remove()
    test_bwd.remove()


def _test_module_forward_preforward_hook_removable(test_case):
    module = nn.Sigmoid()

    def removable_hook(m, input):
        nonlocal handle
        handle.remove()
        return input

    def removable_hook_2(m, input):
        nonlocal handle_2
        handle_2.remove()
        return input

    handle = module.register_forward_pre_hook(removable_hook)
    handle_2 = module.register_forward_pre_hook(removable_hook_2)

    # make sure hook register is successful
    test_case.assertEqual(len(handle.hooks_dict_ref()), 2)
    test_case.assertEqual(len(handle_2.hooks_dict_ref()), 2)

    input = flow.randn(2, 2)
    output = module(input)
    test_case.assertTrue(flow.equal(flow.sigmoid(input), output))

    # make sure hook removal is successful
    test_case.assertFalse(handle.id in handle.hooks_dict_ref())
    test_case.assertFalse(handle_2.id in handle.hooks_dict_ref())
    test_case.assertEqual(len(handle.hooks_dict_ref()), 0)
    test_case.assertEqual(len(handle_2.hooks_dict_ref()), 0)


def _test_module_forward_forward_hook_removable(test_case):
    module = nn.Sigmoid()

    def removable_hook(m, input, output):
        nonlocal handle
        handle.remove()
        return output

    def removable_hook_2(m, input, output):
        nonlocal handle_2
        handle_2.remove()
        return output

    handle = module.register_forward_hook(removable_hook)
    handle_2 = module.register_forward_hook(removable_hook_2)

    # make sure hook register is successful
    test_case.assertEqual(len(handle.hooks_dict_ref()), 2)
    test_case.assertEqual(len(handle_2.hooks_dict_ref()), 2)

    input = flow.randn(2, 2)
    output = module(input)
    test_case.assertTrue(flow.equal(flow.sigmoid(input), output))

    # make sure hook removal is successful
    test_case.assertFalse(handle.id in handle.hooks_dict_ref())
    test_case.assertFalse(handle_2.id in handle.hooks_dict_ref())
    test_case.assertEqual(len(handle.hooks_dict_ref()), 0)
    test_case.assertEqual(len(handle_2.hooks_dict_ref()), 0)


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

    @flow.unittest.skip_unless_1n1d()
    def test_module_delattr(test_case):
        class ConvBNModule(nn.Module):
            def __init__(self):
                super(ConvBNModule, self).__init__()
                self.conv = nn.Conv2d(1, 2, 1, 1)
                self.bn = nn.BatchNorm2d(2)

            def forward(self, x):
                return self.bn(self.conv(x))

        m = ConvBNModule()
        delattr(m, "bn")

    @flow.unittest.skip_unless_1n1d()
    def test_hooks_register(test_case):
        for hook in ["register_backward_hook", "register_full_backward_hook"]:
            _test_hooks(test_case, hook)
        _test_module_forward_preforward_hook_removable(test_case)
        _test_module_forward_forward_hook_removable(test_case)

    @flow.unittest.skip_unless_1n1d()
    def test_register_state_dict_hook_hook(test_case):
        destination_check = None

        def state_dict_hook(module, destination, prefix, local_metadata):
            for submodule_name, submodule in module.named_modules():
                for attr_name, attr in submodule.__dict__.items():
                    if isinstance(attr, torch.Tensor):
                        mod_prefix = prefix + submodule_name
                        key = mod_prefix + ("." if mod_prefix else "") + attr_name
                        destination[key] = attr
            nonlocal destination_check
            destination_check = destination

        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
                self._register_state_dict_hook(state_dict_hook)

            def forward(self, x):
                x = self.linear(x)
                return x

        m = CustomModule()
        test_case.assertEqual(destination_check, None)
        state_dict = m.state_dict()
        test_case.assertEqual(destination_check, state_dict)

    @flow.unittest.skip_unless_1n1d()
    def test_full_backward_hook(test_case):
        hook_triggered = False

        def hook(_, grad_input, grad_output):
            nonlocal hook_triggered
            hook_triggered = True
            test_case.assertEqual(len(grad_input), 1)
            test_case.assertEqual(len(grad_output), 1)
            test_case.assertTrue(np.array_equal(grad_input[0].numpy(), [1, 0]))
            test_case.assertTrue(np.array_equal(grad_output[0].numpy(), [1, 1]))

        m = flow.nn.ReLU()
        m.register_full_backward_hook(hook)

        x0 = flow.tensor([1.0, -1], requires_grad=True)
        x = x0 + 1
        y = m(x)
        y.sum().backward()
        test_case.assertTrue(hook_triggered)
        test_case.assertTrue(np.array_equal(x0.grad, [1, 0]))

    @flow.unittest.skip_unless_1n1d()
    def test_full_backward_hook_with_return_value(test_case):
        hook_triggered = False

        def hook(_, grad_input, grad_output):
            nonlocal hook_triggered
            hook_triggered = True
            test_case.assertEqual(len(grad_input), 1)
            test_case.assertEqual(len(grad_output), 1)
            test_case.assertTrue(np.array_equal(grad_input[0].numpy(), [1, 0]))
            test_case.assertTrue(np.array_equal(grad_output[0].numpy(), [1, 1]))
            return (flow.tensor([1, 1]),)

        m = flow.nn.ReLU()
        m.register_full_backward_hook(hook)

        x0 = flow.tensor([1.0, -1], requires_grad=True)
        x = x0 + 1
        y = m(x)
        y.sum().backward()
        test_case.assertTrue(hook_triggered)
        test_case.assertTrue(np.array_equal(x0.grad, [1, 1]))


if __name__ == "__main__":
    unittest.main()
