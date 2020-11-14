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
import collections
import os
from collections import OrderedDict

import numpy as np
import oneflow as flow
import torch
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import oneflow.typing as oft

unfold_confs = [
    {
        "x_shape": (1, 1, 6, 6),
        "ksize": 1,
        "strides": 1,
        "dilation_rate": 1,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 3, 7, 7),
        "ksize": 3,
        "strides": 2,
        "dilation_rate": 1,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 5, 6, 6),
        "ksize": 2,
        "strides": 2,
        "dilation_rate": 2,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 7, 5, 5),
        "ksize": 3,
        "strides": 1,
        "dilation_rate": 1,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 3, 3, 3),
        "ksize": 1,
        "strides": 2,
        "dilation_rate": 3,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 1, 9, 9),
        "ksize": 2,
        "strides": 3,
        "dilation_rate": 2,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 1, 6, 6),
        "ksize": 2,
        "strides": 1,
        "dilation_rate": 1,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 3, 7, 7),
        "ksize": 3,
        "strides": 1,
        "dilation_rate": 1,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 5, 6, 6),
        "ksize": 2,
        "strides": 1,
        "dilation_rate": 2,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 7, 5, 5),
        "ksize": 3,
        "strides": 1,
        "dilation_rate": 1,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 3, 3, 3),
        "ksize": 1,
        "strides": 1,
        "dilation_rate": 3,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 1, 9, 9),
        "ksize": 2,
        "strides": 1,
        "dilation_rate": 2,
        "padding": "SAME",
        "data_format": "NCHW",
    },
]


def _GetSequence(value, n, name):
    """Formats value from input"""
    if value is None:
        value = [1]
    elif not isinstance(value, collections.Sized):
        value = [value]

    current_n = len(value)
    if current_n == 1:
        return list(value * n)
    elif current_n == n:
        return list(value)
    else:
        raise ValueError(
            "{} should be of length 1 or {} but was {}".format(name, n, current_n)
        )


@flow.unittest.skip_unless_1n1d()
class TestUnfoldPadding(flow.unittest.TestCase):
    def test_unfold(_):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu"]
        arg_dict["unfold_conf"] = unfold_confs
        arg_dict["data_type"] = ["float32"]

        for case in GenArgList(arg_dict):
            (device_type, unfold_conf, data_type) = case
            x_shape = unfold_conf["x_shape"]
            ksize = unfold_conf["ksize"]
            strides = unfold_conf["strides"]
            dilation_rate = unfold_conf["dilation_rate"]
            padding = unfold_conf["padding"]
            data_format = unfold_conf["data_format"]
            flow.clear_default_session()

            dim = len(x_shape) - 2
            # TODO: not implement currently
            if dim == 3 or data_format == "NHWC":
                continue

            ksize = _GetSequence(ksize, dim, "ksize")
            strides = _GetSequence(strides, dim, "strides")
            dilation_rate = _GetSequence(dilation_rate, dim, "dilation_rate")
            padding_torch = list(0 for _ in range(dim))
            in_dhw = list(x_shape)[-dim :]

            print(case)
            valid_case = True
            if padding == 'SAME':
                out_dhw = in_dhw.copy()
                for i in range(dim):
                    padding_torch[i] = (ksize[i] - 1) * dilation_rate[i]
                    if strides[i] != 1 or padding_torch[i] % 2 != 0:
                        valid_case = False
                    padding_torch[i] //= 2

            # Random inputs
            #x = np.arange(np.prod(x_shape)).astype(type_name_to_np_type[data_type]).reshape(*x_shape)
            x = np.random.randn(*x_shape).astype(type_name_to_np_type[data_type])

            # torch results
            x_torch = torch.tensor(x, requires_grad=True, device=device_type, dtype=torch.float)
            model = torch.nn.Unfold(ksize, stride=strides, padding=tuple(padding_torch), dilation=dilation_rate)
            y_torch = model(x_torch)
            z = y_torch.sum()
            z.backward()

            def assert_grad(b):
                if valid_case:
                    assert np.allclose(x_torch.grad.numpy(), b.numpy()), (
                        case,
                        x_torch.grad.numpy(),
                        b.numpy(),
                    )
                else:
                    print('not valid: ', case)

            # 1F results
            dtype = type_name_to_flow_type[data_type]

            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)

            tensor_def = None
            tensor_def = oft.Numpy.Placeholder

            @flow.global_function(type="train", function_config=func_config)
            def unfold_job(x: tensor_def(x_shape, dtype=dtype)):
                v = flow.get_variable(
                    "x",
                    shape=x_shape,
                    dtype=dtype,
                    initializer=flow.constant_initializer(0),
                    trainable=True,
                )
                v = flow.cast_to_current_logical_view(v)
                flow.watch_diff(v, assert_grad)
                x += v
                with flow.scope.placement(device_type, "0:0"):
                    unfold_f = getattr(flow.nn, "unfold{}d".format(dim))

                    padding = unfold_conf["padding"]
                    if padding == "SAME":
                        padding = "SAME_UPPER"
                    y = unfold_f(
                        x,
                        ksize=ksize,
                        strides=strides,
                        dilation_rate=dilation_rate,
                        padding=padding,
                        data_format=data_format,
                    )
                    flow.optimizer.SGD(
                        flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
                    ).minimize(y)
                return y

            y = unfold_job(x).get()
            y_ndarray = y.numpy()
            if valid_case:
                assert y_ndarray.shape == y_torch.detach().numpy().shape, (
                    y_ndarray.shape,
                    y_torch.detach().numpy().shape,
                )
                assert np.allclose(y_ndarray, y_torch.detach().numpy(), rtol=1e-5, atol=1e-5), (
                    case,
                    y_ndarray.shape,
                    y_ndarray,
                    y_torch.detach().numpy(),
                )
            else:
                print('not valid: ', case)


if __name__ == "__main__":
    unittest.main()
