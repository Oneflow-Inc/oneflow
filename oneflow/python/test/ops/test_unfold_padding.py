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
        "x_shape": (2, 3, 6, 6),
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
        "x_shape": (4, 1, 9, 9),
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
        "x_shape": (4, 1, 9, 9),
        "ksize": 2,
        "strides": 1,
        "dilation_rate": 2,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 3, 3, 3),
        "ksize": 1,
        "strides": 1,
        "dilation_rate": 3,
        "padding": (0, 0),
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 2, 8, 8),
        "ksize": 3,
        "strides": 1,
        "dilation_rate": 2,
        "padding": 2,
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 2, 8, 8),
        "ksize": 3,
        "strides": 1,
        "dilation_rate": 2,
        "padding": (2, 2),
        "data_format": "NCHW",
    },
    {
        "x_shape": (1, 2, 8, 8),
        "ksize": 3,
        "strides": 1,
        "dilation_rate": 2,
        "padding": ((1, 1), (2, 2)),
        "data_format": "NCHW",
    },
    {
        "x_shape": (3, 2, 8, 8),
        "ksize": 3,
        "strides": 1,
        "dilation_rate": 2,
        "padding": [(1, 2), (1, 2)],
        "data_format": "NCHW",
    },
]


unfold_confs_1n2d = [
    {
        "x_shape": (2, 2, 6, 6),
        "ksize": 1,
        "strides": 1,
        "dilation_rate": 1,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (2, 5, 6, 6),
        "ksize": 2,
        "strides": 2,
        "dilation_rate": 2,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (4, 3, 3, 3),
        "ksize": 1,
        "strides": 2,
        "dilation_rate": 3,
        "padding": "VALID",
        "data_format": "NCHW",
    },
    {
        "x_shape": (2, 3, 7, 7),
        "ksize": 3,
        "strides": 1,
        "dilation_rate": 1,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (4, 7, 5, 5),
        "ksize": 3,
        "strides": 1,
        "dilation_rate": 1,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (4, 3, 9, 9),
        "ksize": 2,
        "strides": 2,
        "dilation_rate": 2,
        "padding": "SAME",
        "data_format": "NCHW",
    },
    {
        "x_shape": (6, 2, 8, 8),
        "ksize": 3,
        "strides": 1,
        "dilation_rate": 2,
        "padding": 2,
        "data_format": "NCHW",
    },
    {
        "x_shape": (3, 2, 8, 8),
        "ksize": 3,
        "strides": 1,
        "dilation_rate": 2,
        "padding": ((1, 1), (2, 2)),
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


def _GetTorchPadding(padding, dim, in_dhw, ksize, strides, dilation_rate):
    valid_case = True
    torch_padding = [0 for _ in range(dim)]
    if isinstance(padding, int):
        torch_padding = _GetSequence(padding, 2, "padding")
    elif isinstance(padding, (list, tuple)):
        for i in range(dim):
            if isinstance(padding[i], int):
                torch_padding[i] = padding[i]
            elif isinstance(padding[i], (list, tuple)):
                if padding[i][0] != padding[i][1]:
                    valid_case = False
                torch_padding[i] = padding[i][0]
    elif isinstance(padding, str):
        padding = padding.upper()
        padding = "SAME_LOWER" if padding == "SAME" else padding
        assert padding.upper() in ["VALID", "SAME_LOWER", "SAME_UPPER"]
        if padding.startswith("SAME"):
            out_dhw = in_dhw.copy()
            for i in range(dim):
                torch_padding[i] = (ksize[i] - 1) * dilation_rate[i]
                if strides[i] != 1 or torch_padding[i] % 2 != 0:
                    valid_case = False
                torch_padding[i] //= 2
    else:
        raise NotImplementedError
    return valid_case, tuple(torch_padding)


def _compare_with_samples(case):
    (device_type, device_count, machine_ids, unfold_conf, data_type) = case
    torch_device = torch.device("cuda" if device_type == "gpu" else "cpu")
    x_shape = unfold_conf["x_shape"]
    ksize = unfold_conf["ksize"]
    strides = unfold_conf["strides"]
    dilation_rate = unfold_conf["dilation_rate"]
    padding = unfold_conf["padding"]
    data_format = unfold_conf["data_format"]
    flow.clear_default_session()

    dim = len(x_shape) - 2
    ksize = _GetSequence(ksize, dim, "ksize")
    strides = _GetSequence(strides, dim, "strides")
    dilation_rate = _GetSequence(dilation_rate, dim, "dilation_rate")
    in_dhw = list(x_shape)[-dim:]
    valid_case, torch_padding = _GetTorchPadding(
        padding, 2, in_dhw, ksize, strides, dilation_rate
    )

    # Random inputs
    x = np.random.randn(*x_shape).astype(type_name_to_np_type[data_type])

    # torch results
    x_torch = torch.tensor(
        x, requires_grad=True, device=torch_device, dtype=torch.float
    )
    model = torch.nn.Unfold(
        ksize, stride=strides, padding=torch_padding, dilation=dilation_rate,
    )
    model.to(torch_device)
    y_torch = model(x_torch)
    z = y_torch.sum()
    z.backward()

    def assert_grad(b):
        if not valid_case:
            print("not valid: ", case)
            return
        x_torch_grad_numpy = x_torch.grad.cpu().numpy()
        assert np.allclose(x_torch_grad_numpy, b.numpy()), (
            case,
            x_torch_grad_numpy,
            b.numpy(),
        )

    # 1F results
    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_count)
    else:
        flow.config.gpu_device_num(device_count)

    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))
    func_config.default_logical_view(flow.scope.consistent_view())

    dtype = type_name_to_flow_type[data_type]
    tensor_def = oft.Numpy.Placeholder

    @flow.global_function(type="train", function_config=func_config)
    def unfold_job(x: tensor_def(x_shape, dtype=dtype)):
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=dtype,
                initializer=flow.constant_initializer(0),
                trainable=True,
            )
            v = flow.cast_to_current_logical_view(v)
            x += v

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
        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0,
            ).minimize(y)

        flow.watch_diff(v, assert_grad)
        return y

    y = unfold_job(x).get()
    y_ndarray = y.numpy()
    if not valid_case:
        return

    y_torch_numpy = y_torch.detach().cpu().numpy()
    assert y_ndarray.shape == y_torch_numpy.shape, (
        y_ndarray.shape,
        y_torch_numpy.shape,
    )
    assert np.allclose(y_torch_numpy, y_ndarray, rtol=1e-5, atol=1e-5), (
        case,
        y_ndarray.shape,
        y_torch_numpy,
        y_ndarray,
    )


@flow.unittest.skip_unless_1n1d()
class TestUnfoldPadding1n1d(flow.unittest.TestCase):
    def test_unfold_cpu(_):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu"]
        arg_dict["device_count"] = [1]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["unfold_conf"] = unfold_confs
        arg_dict["data_type"] = ["float32", "double"]

        for case in GenArgList(arg_dict):
            _compare_with_samples(case)

    def test_unfold_gpu_1n1d(_):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["device_count"] = [1]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["unfold_conf"] = unfold_confs
        arg_dict["data_type"] = ["float32", "double"]

        for case in GenArgList(arg_dict):
            _compare_with_samples(case)


@flow.unittest.skip_unless_1n2d()
class TestUnfoldPadding1n2d(flow.unittest.TestCase):
    def test_unfold_gpu_1n2d(_):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["device_count"] = [2]
        arg_dict["machine_ids"] = ["0:0-1"]
        arg_dict["unfold_conf"] = unfold_confs_1n2d
        arg_dict["data_type"] = ["float32", "double"]

        for case in GenArgList(arg_dict):
            _compare_with_samples(case)


if __name__ == "__main__":
    unittest.main()
