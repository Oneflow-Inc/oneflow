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
from collections import OrderedDict

import test_global_storage
from test_util import GenArgList, type_name_to_flow_type


def compare_with_tensorflow(device_type, device_num, in_shape, data_type, degree):
    assert device_type in ["cpu", "gpu"]
    assert data_type in ["float32", "double"]
    flow_data_type = type_name_to_flow_type[data_type]
    flow.clear_default_session()
    flow.config.gpu_device_num(device_num)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow_data_type)

    @flow.global_function(function_config=func_config)
    def PolyValJob():
        with flow.scope.placement(device_type, "0:0-{}".format(device_num - 1)):
            x = flow.get_variable(
                "x",
                shape=in_shape,
                dtype=flow_data_type,
                initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                trainable=True,
            )
            flow.watch(x, test_global_storage.Setter("x"))
            coeffs = []
            for i in range(degree):
                tmp = flow.get_variable(
                    name="coeff_%d" % i,
                    shape=in_shape,
                    dtype=flow_data_type,
                    initializer=flow.random_uniform_initializer(minval=-10, maxval=10),
                )
                coeffs.append(tmp)
                flow.watch(tmp, test_global_storage.Setter("coeff_%d" % i))
            return flow.math.polyval(coeffs, x)

    # OneFlow
    check_point = flow.train.CheckPoint()
    check_point.init()
    of_out = PolyValJob().get().numpy()
    # numpy
    x = test_global_storage.Get("x")
    coeffs = []
    for i in range(degree):
        coeffs.append(test_global_storage.Get("coeff_%d" % i))
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
    arg_dict["in_shape"] = [(1,), (2, 1), (1, 2), (2, 2)]
    arg_dict["data_type"] = ["float32", "double"]
    arg_dict["degree"] = [0, 1, 3, 5]
    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class TestPolyval1n1d(flow.unittest.TestCase):
    def test_polyval(test_case):
        for arg in gen_arg_list("1n1d"):
            compare_with_tensorflow(*arg)


@flow.unittest.skip_unless_1n2d()
class TestPolyval1n2d(flow.unittest.TestCase):
    def test_polyval(test_case):
        for arg in gen_arg_list("1n2d"):
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
