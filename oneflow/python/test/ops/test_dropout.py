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
import os
import shutil
from collections import OrderedDict

import numpy as np
import oneflow as flow
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type


def of_run(device_type, x_shape, data_type, rate, seed):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()

    if data_type == "float16":
        dtype = flow.float
    else:
        dtype = type_name_to_flow_type[data_type]

    @flow.global_function(type="train", function_config=func_config)
    def DropoutJob():
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=dtype,
                initializer=flow.random_uniform_initializer(minval=-1, maxval=1),
                trainable=True,
            )
            if data_type == "float16":
                x = flow.cast(flow.cast(x, flow.float16), dtype)
                of_out = flow.cast(
                    flow.nn.dropout(
                        flow.cast(x, flow.float16), rate=rate, seed=seed, name="dropout"
                    ),
                    dtype,
                )
            else:
                of_out = flow.nn.dropout(x, rate=rate, seed=seed, name="dropout")
            loss = flow.math.square(of_out)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)

            flow.watch(x, test_global_storage.Setter("x"))
            flow.watch_diff(x, test_global_storage.Setter("x_diff"))
            flow.watch(of_out, test_global_storage.Setter("out"))
            flow.watch_diff(of_out, test_global_storage.Setter("out_diff"))

            return loss

    # OneFlow
    of_out = DropoutJob().get()

    of_out = test_global_storage.Get("out")
    out_diff = test_global_storage.Get("out_diff")
    assert np.allclose(
        [1 - np.count_nonzero(of_out) / of_out.size], [rate], atol=rate / 5
    )
    x = test_global_storage.Get("x")
    x_diff = test_global_storage.Get("x_diff")
    out_scale = of_out[np.where(of_out != 0)] / x[np.where(of_out != 0)]
    diff_scale = x_diff[np.where(of_out != 0)] / out_diff[np.where(of_out != 0)]
    assert np.allclose(out_scale, 1.0 / (1.0 - rate), atol=1e-5)
    assert np.allclose(diff_scale, 1.0 / (1.0 - rate), atol=1e-5)


def of_run_module(device_type, x_shape, data_type, rate, seed):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    dtype = type_name_to_flow_type[data_type]

    @flow.global_function(type="train", function_config=func_config)
    def DropoutJob() -> flow.typing.Numpy:
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=dtype,
                initializer=flow.ones_initializer(),
                trainable=True,
            )
            of_out = flow.nn.dropout(x, rate=rate, seed=seed, name="dropout")
            loss = flow.math.square(of_out)
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
            ).minimize(loss)
            return of_out

    of_out = DropoutJob()
    of_out2 = DropoutJob()

    return of_out, of_out2


@flow.unittest.skip_unless_1n1d()
class TestDropout(flow.unittest.TestCase):
    def test_dropout(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["x_shape"] = [(100, 100, 10, 20)]
        arg_dict["data_type"] = ["float32", "double", "float16"]
        arg_dict["rate"] = [0.75]
        arg_dict["seed"] = [12345, None]
        for arg in GenArgList(arg_dict):
            if arg[0] == "cpu" and arg[2] == "float16":
                continue
            of_run(*arg)

    def test_dropout_module(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cpu", "gpu"]
        arg_dict["x_shape"] = [(2, 2, 2, 2)]
        arg_dict["data_type"] = ["float32"]
        arg_dict["rate"] = [0.75]
        arg_dict["seed"] = [12345]

        literals = {
            "cpu": [
                np.array(
                    [
                        4.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        4.0,
                        0.0,
                        0.0,
                        4.0,
                    ]
                ),
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                    ]
                ),
            ],
            "gpu": [
                np.array(
                    [
                        4.0,
                        4.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ),
                np.array(
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        4.0,
                        0.0,
                    ]
                ),
            ],
        }

        for arg in GenArgList(arg_dict):
            of_out_a, of_out_b = of_run_module(*arg)
            test_case.assertEqual(
                (np.abs(literals[arg[0]][0] - of_out_a.flatten()) < 10e-7).all(), True
            )
            test_case.assertEqual(
                (np.abs(literals[arg[0]][1] - of_out_b.flatten()) < 10e-7).all(), True
            )


if __name__ == "__main__":
    unittest.main()
