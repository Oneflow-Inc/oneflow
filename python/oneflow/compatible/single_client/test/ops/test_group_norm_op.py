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
import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp


def getGroupNormOutAndGrad(input, gout, num_groups, eps):
    assert len(input.shape) == len(gout.shape)
    assert len(input.shape) >= 3
    channel = input.shape[1]
    assert channel % num_groups == 0
    orig_shape = input.shape
    input_reshape_to_1d = np.reshape(input, (input.shape[0], num_groups, -1))
    gout_reshape_to_1d = np.reshape(gout, (gout.shape[0], num_groups, -1))
    gamma = np.ones((1, channel, 1), dtype=np.float32)
    mean_np = np.mean(input_reshape_to_1d, axis=2, keepdims=True)
    in_sub_mean = input_reshape_to_1d - mean_np
    var_np = np.mean(np.square(in_sub_mean), axis=2, keepdims=True)
    invar_np = 1.0 / np.sqrt(var_np + eps)
    out_np = np.reshape(in_sub_mean * invar_np, (input.shape[0], channel, -1)) * gamma
    gvar = (
        np.reshape(
            gout_reshape_to_1d * in_sub_mean * -0.5 * np.power(var_np + eps, -1.5),
            (gout.shape[0], channel, -1),
        )
        * gamma
    )
    gvar = np.reshape(gvar, (gout.shape[0], num_groups, -1))
    gvar = np.sum(gvar, axis=2, keepdims=True)
    gmean = np.reshape(gout_reshape_to_1d, (gout.shape[0], channel, -1)) * gamma
    gmean = np.sum(
        np.reshape(gmean, (gout.shape[0], num_groups, -1)), axis=2, keepdims=True
    )
    gmean *= -invar_np
    scale = 1.0 / input_reshape_to_1d.shape[2]
    tmp = scale * np.sum(-2.0 * in_sub_mean, axis=2, keepdims=True) * gvar
    gmean += tmp
    gin_np = (
        np.reshape(
            gout_reshape_to_1d * invar_np
            + gvar * scale * 2.0 * in_sub_mean
            + gmean * scale,
            (input.shape[0], channel, -1),
        )
        * gamma
    )
    return (np.reshape(out_np, list(orig_shape)), np.reshape(gin_np, list(orig_shape)))


def _compare_group_norm_nd_with_np(
    input_shape, device_type, machine_ids, device_counts, num_groups, eps, affine
):
    assert device_type in ["cpu", "gpu"]
    assert len(input_shape) >= 3 and len(input_shape) <= 5
    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)
    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))
    input = np.random.random(size=input_shape).astype(np.float32)
    gout = np.random.random(size=input_shape).astype(np.float32)
    (out_np, gin_np) = getGroupNormOutAndGrad(input, gout, num_groups, eps)

    def assert_prediction_grad(gin_of: tp.Numpy):
        assert np.allclose(gin_of, gin_np, atol=1e-05)

    @flow.global_function(type="train", function_config=func_config)
    def groupNormJob(
        of_input: tp.Numpy.Placeholder(shape=input.shape),
        multipler: tp.Numpy.Placeholder(shape=input.shape),
    ) -> tp.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                shape=of_input.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
                name="v",
            )
            x_var = of_input + v
            flow.watch_diff(x_var, assert_prediction_grad)
        out = flow.nn.GroupNorm(x_var, num_groups=num_groups, eps=eps, affine=True)
        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [0.001]), momentum=0
            ).minimize(out * multipler)
        return out

    of_out = groupNormJob(input, gout)
    assert np.allclose(of_out, out_np, atol=1e-05)


@flow.unittest.skip_unless_1n1d()
class TestGroupNormND1n1d(flow.unittest.TestCase):
    def test_group_norm(test_case):
        arg_dict = OrderedDict()
        arg_dict["input_shape"] = [(4, 8, 32, 32)]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_counts"] = [1]
        arg_dict["num_groups"] = [4, 8]
        arg_dict["eps"] = [0.001]
        arg_dict["affine"] = [True, False]
        for arg in GenArgList(arg_dict):
            _compare_group_norm_nd_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class TestGroupNormND1n2d(flow.unittest.TestCase):
    def test_group_norm(test_case):
        arg_dict = OrderedDict()
        arg_dict["input_shape"] = [(4, 8, 32, 32)]
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["machine_ids"] = ["0:0-1"]
        arg_dict["device_counts"] = [2]
        arg_dict["num_groups"] = [4, 8]
        arg_dict["eps"] = [0.001]
        arg_dict["affine"] = [True, False]
        for arg in GenArgList(arg_dict):
            _compare_group_norm_nd_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
