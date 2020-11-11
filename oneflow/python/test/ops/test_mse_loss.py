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
import oneflow as flow
import numpy as np
import oneflow.typing as tp
from test_util import GenArgList
import unittest
from collections import OrderedDict
from typing import List


def _compare_mseloss_with_np(input, target, device_type, machine_ids, device_counts):
    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    flow.env.init()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_mseloss(
        of_input: tp.Numpy.Placeholder(shape=input.shape),
        of_target: tp.Numpy.Placeholder(shape=target.shape),
    ) -> List[tp.Numpy]:
        x_var = flow.get_variable(
            shape=input.shape,
            dtype=flow.float32,
            initializer=flow.ones_initializer(),
            name="x_var",
        )

        with flow.scope.placement(device_type, machine_ids):
            mseloss = flow.nn.MSELoss(
                of_input, of_target, reduction="none", name="of_mseloss"
            )
            mseloss_mean = flow.nn.MSELoss(
                of_input, of_target, reduction="mean", name="of_mseloss_reduce_mean"
            )
            mseloss_sum = flow.nn.MSELoss(
                of_input, of_target, reduction="sum", name="of_mseloss_reduce_sum"
            )

            out = mseloss + x_var
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(out)

        return [mseloss, mseloss_mean, mseloss_sum]

    def np_mseloss(np_input, np_target):
        np_mse = np.square(np_target - np_input)
        np_mse_mean = np.mean(np_mse)
        np_mse_sum = np.sum(np_mse)

        return [np_mse, np_mse_mean, np_mse_sum]

    of_out_mseloss = oneflow_mseloss(input, target)
    np_out_mseloss = np_mseloss(input, target)

    assert np.array_equal(of_out_mseloss[0], np_out_mseloss[0])

    for i in range(1, len(np_out_mseloss)):
        # TODO: Should I change to use np.allclose?
        # There may have some numerical error in float value

        # Numpy return a scalar(no shape), but oneflow return a N-D tensor.
        # I need to get it by using index [0]
        assert np.array_equal(of_out_mseloss[i][0], np_out_mseloss[i])


@flow.unittest.skip_unless_1n1d()
class Testl1loss1n1d(flow.unittest.TestCase):
    def test_mseloss_cpu(test_case):
        arg_dict = OrderedDict()
        arg_dict["input"] = [
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        ]
        arg_dict["target"] = [
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.float32)
        ]
        arg_dict["device_type"] = ["cpu"]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_counts"] = [1]
        for arg in GenArgList(arg_dict):
            _compare_mseloss_with_np(*arg)

    def test_mseloss_gpu(test_case):
        arg_dict = OrderedDict()
        arg_dict["input"] = [
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        ]
        arg_dict["target"] = [
            np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]]).astype(np.float32)
        ]
        arg_dict["device_type"] = ["gpu"]
        arg_dict["machine_ids"] = ["0:0"]
        arg_dict["device_counts"] = [1]
        for arg in GenArgList(arg_dict):
            _compare_mseloss_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Testrange1n2d(flow.unittest.TestCase):
    def test_mseloss_gpu_1n2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["input"] = [
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        ]
        arg_dict["target"] = [
            np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]]).astype(np.float32)
        ]
        arg_dict["device_type"] = ["gpu"]
        arg_dict["machine_ids"] = ["0:0-1"]
        arg_dict["device_counts"] = [2]
        for arg in GenArgList(arg_dict):
            _compare_mseloss_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
