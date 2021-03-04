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
import oneflow.typing as tp
from collections import OrderedDict

from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


def compare_with_numpy(device_type, device_num, in_shape, data_type, coeffs):
    assert device_type in ["cpu", "gpu"]
    assert data_type in ["float32", "double"]
    flow_data_type = type_name_to_flow_type[data_type]
    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_num)
    else:
        flow.config.gpu_device_num(device_num)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow_data_type)
    func_config.default_placement_scope(
        flow.scope.placement(device_type, "0:0-{}".format(device_num - 1))
    )
    func_config.default_logical_view(flow.scope.consistent_view())

    x = (np.random.random(in_shape) * 100).astype(type_name_to_np_type[data_type])

    def np_polyval_grad(coeffs, x):
        coeffs_len = len(coeffs)
        coeffs_diff = [(coeffs_len - i - 1) * coeffs[i] for i in range(coeffs_len - 1)]
        np_x_diff = np.polyval(coeffs_diff, x)
        return np_x_diff

    def assert_prediction_grad(blob: tp.Numpy):
        np_x_diff = np_polyval_grad(coeffs, x)
        assert np.allclose(blob, np_x_diff, rtol=1e-5, atol=1e-5)

    @flow.global_function(type="train", function_config=func_config)
    def PolyValJob(x: tp.Numpy.Placeholder(shape=in_shape)):
        with flow.scope.placement(device_type, "0:0"):
            x += flow.get_variable(
                name="x",
                shape=in_shape,
                dtype=flow_data_type,
                initializer=flow.zeros_initializer(),
                trainable=True,
            )
        flow.watch_diff(x, assert_prediction_grad)
        out = flow.math.polyval(coeffs, x)
        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(out)

        return out

    of_out = PolyValJob(x).get().numpy()
    np_out = np.polyval(coeffs, x)
    assert np.allclose(of_out, np_out, rtol=1e-5, atol=1e-5)


def gen_arg_list(type):
    arg_dict = OrderedDict()
    if type == "1n2d":
        arg_dict["device_type"] = ["gpu"]
        arg_dict["device_num"] = [2]
    else:
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["device_num"] = [1]
    arg_dict["in_shape"] = [(2, 3)]
    arg_dict["data_type"] = ["float32"]
    arg_dict["coeffs"] = [[1.0, 2.0], [1.0, 2.0, 3.0]]
    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestPolyval1n1d(flow.unittest.TestCase):
    def test_polyval(test_case):
        for arg in gen_arg_list("1n1d"):
            compare_with_numpy(*arg)


@flow.unittest.skip_unless_1n2d()
class TestPolyval1n2d(flow.unittest.TestCase):
    def test_polyval(test_case):
        for arg in gen_arg_list("1n2d"):
            compare_with_numpy(*arg)


if __name__ == "__main__":
    unittest.main()
