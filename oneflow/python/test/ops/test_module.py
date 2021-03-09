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

import numpy as np

import oneflow as flow
import oneflow.typing as tp


def _wrapper(func):
    def wrapped_func(*args):
        args = list(args)
        for i in range(len(args)):
            arg = args[i]
            if isinstance(arg, flow.Tensor):
                if not arg.is_determined:
                    arg.determine()
                args[i] = arg._local_or_consistent_tensor

        res = func(*args)

        tensor = flow.Tensor(shape=res.shape, dtype=res.dtype)
        tensor._local_or_consistent_tensor = res
        tensor._undetermined_tensor = None

        return tensor

    return wrapped_func
    

class Relu(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self._op = (
            flow.builtin_op("relu").Name("relu").Input("in").Output("out").Build()
        )

    
    @_wrapper
    def forward(self, x):
        res = self._op(x)[0]
        return res


_size_2_t = Union[int, Tuple[int, int]]

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)

class Conv2d(flow.nn.Module):
    def __init__(self, 
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
            ):
        super().__init__()

        assert padding_mode == "zeros"
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.weight = flow.nn.Parameter(flow.Tensor((out_channels, in_channels // groups, *kernel_size), dtype=flow.float32))
        self._op = (
            flow.builtin_op("conv2d").Name("conv2d")
            .Input("in")
            .Input("weight")
            .Attr('filters', out_channels)
            .Attr("padding_before", padding)
            .Attr("strides", stride)
            .Attr("kernel_size", kernel_size)
            .Attr("dilation_rate", dilation)
            .Attr("groups", groups)
            .Attr("data_format", 'channels_first')
            .Output("out").Build()
        )
    
    @_wrapper
    def forward(self, x):
        if not self.weight.is_determined:
            self.weight.determine()
        res = self._op(x, self.weight._local_or_consistent_tensor)[0]
        return res


_counter = 0


def get_var_helper(shape):
    global _counter
    var = flow.get_variable(
        "x_" + str(_counter), shape=shape, initializer=flow.kaiming_initializer(shape)
    )
    _counter += 1
    return var


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_conv2d(test_case):
        conv2d = Conv2d(3, 3, 3)

        @flow.global_function()
        def job() -> None:
            x = flow.Tensor((1, 3, 4, 5), dtype=flow.float32)
            global y
            y = conv2d(x)
            print(conv2d.weight.numpy())
            print(y.numpy())

        job()

    def test_relu(test_case):
        relu = Relu()

        @flow.global_function()
        def job() -> None:
            x = flow.Tensor((2, 3), dtype=flow.float32)
            global y
            y = relu(x)
            print(y.numpy())

        job()

        print(y)

    def test_load_state_dict(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = flow.nn.Parameter(flow.Tensor((2, 3), dtype=flow.float32))

            def forward(self, x):
                return self.w

        m = CustomModule()

        @flow.global_function()
        def job() -> None:
            x = flow.Tensor((2, 3), dtype=flow.float32)
            global y
            y = m(x).numpy()

        job()
        ones = np.ones((2, 3), dtype=np.float32)
        m.load_state_dict({"w": ones})
        job()
        test_case.assertTrue(np.array_equal(y, ones))

    def test_state_dict(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self, param1, param2):
                super().__init__()
                self.param1 = param1
                self.param2 = param2

        tensor0 = flow.nn.Parameter(flow.Tensor((2, 3), dtype=flow.float32))
        tensor1 = flow.nn.Parameter(flow.Tensor((2, 3), dtype=flow.float32))
        sub_module = CustomModule(tensor0, tensor1)
        m = CustomModule(tensor1, sub_module)

        state_dict = m.state_dict()
        test_case.assertEqual(
            state_dict,
            {"param2.param1": tensor0, "param2.param2": tensor1, "param1": tensor1},
        )

    def test_parameter(test_case):
        shape = (3, 4)
        t = flow.Tensor(shape, dtype=flow.float32)
        p = flow.nn.Parameter(t)
        test_case.assertEqual(type(p), flow.nn.Parameter)
        test_case.assertEqual(tuple(p.shape), shape)

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

    # def test_forward_with_variable(test_case):
    #     class AddTo(flow.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.w = flow.nn.Parameter(flow.Tensor(2, 3))
    #
    #         def forward(self, x):
    #             return x + self.w()
    #
    #     @flow.global_function()
    #     def job() -> Tuple[tp.Numpy, tp.Numpy]:
    #         x = get_var_helper((2, 3))
    #         m = AddTo()
    #         return m(x), m.w() + x
    #
    #     res1, res2 = job()
    #     test_case.assertTrue(np.array_equal(res1, res2))

    def test_forward_with_sbp(test_case):
        class AddTo(flow.nn.Module):
            def __init__(self, w):
                super().__init__()
                self.w = w

            def forward(self, x, *args):
                return x + self.w

        @flow.global_function()
        def job() -> Tuple[tp.Numpy, tp.Numpy]:
            w = get_var_helper((2, 3))
            x = get_var_helper((2, 3))
            m = AddTo(w)
            m.input_configs[0] = flow.distribute.split(0)
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

        param0 = flow.nn.Parameter(flow.Tensor((2, 3), dtype=flow.float32))
        param1 = flow.nn.Parameter(flow.Tensor((2, 3), dtype=flow.float32))
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

    @unittest.skip("it is not related to module itself")
    def test_consistent_mirrored(test_case):
        flow.config.gpu_device_num(flow.unittest.env.device_num())

        @flow.global_function()
        def job():
            x1 = get_var_helper((4, 4))
            x2 = get_var_helper((4, 4))
            x3 = x1 + x2
            x4 = flow.advanced.distribute_split(x3)
            parallel_desc_symbol = flow.current_scope().device_parallel_desc_symbol
            device_tag = parallel_desc_symbol.device_tag
            x_list = []
            parallel_id = 0
            for (
                machine_id,
                device_ids,
            ) in parallel_desc_symbol.machine_id2device_id_list.items():
                for device_id in device_ids:
                    with flow.scope.placement(
                        device_tag, str(machine_id) + ":" + str(device_id)
                    ):
                        x5 = x4[parallel_id]
                        if parallel_id == 1:
                            x6 = x5 + 100
                        else:
                            x6 = flow.identity(x5)
                        print(x6.numpy())
                        x_list.append(x6)
                        parallel_id += 1
            x8 = flow.advanced.distribute_concat(x_list)
            flow.watch(x8, lambda x: print(x.numpy()))

        job()

    @unittest.skip("tensor __add__ is not implemented now")
    def test_x(test_case):
        class CustomModule(flow.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = flow.nn.Parameter(flow.Tensor((2, 3), dtype=flow.float32))

            def forward(self, x):
                return x + self.w

        m = CustomModule()
        print(m.state_dict())

        @flow.global_function()
        def job() -> None:
            x = flow.Tensor((2, 3), dtype=flow.float32)
            print(m(x).numpy())

        job()
        m.load_state_dict({"x_2": np.ones((2, 3), dtype=np.float32)})
        job()

    # TODO: add more tests about module api


if __name__ == "__main__":
    unittest.main()
