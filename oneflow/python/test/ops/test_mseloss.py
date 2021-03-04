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
from typing import Dict
import os


def _compare_mseloss_with_np(
    input_shape, target_shape, device_type, machine_ids, device_counts
):
    input = np.random.random(size=input_shape).astype(np.float32)
    target = np.random.random(size=target_shape).astype(np.float32)

    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()

    def np_mseloss(np_input, np_target):
        np_mse = np.square(np_target - np_input)
        np_mse_mean = np.mean(np_mse)
        np_mse_sum = np.sum(np_mse)

        return {
            "np_mse_loss": np_mse,
            "np_mse_loss_mean": np_mse_mean,
            "np_mse_loss_sum": np_mse_sum,
        }

    def np_mseloss_grad(np_input, np_target):
        elem_cnt = np_input.size
        np_mse_grad_mean = (-2 * (np_target - np_input)) / elem_cnt

        # TODO: if you want to get the grad when the reduction="sum", you can use the follow code
        # np_mse_grad_sum = -2 * (np_target - np_input)

        return {
            "np_mse_grad_mean": np_mse_grad_mean,
        }

    # Use Numpy to compute mseloss
    np_out_mseloss_dict = np_mseloss(input, target)
    # Use Numpy to compute mseloss grad
    np_grad_dict = np_mseloss_grad(input, target)

    def assert_prediction_grad(blob: tp.Numpy):
        # Evaluate the gradient. Here we only test the reduction type == "mean"
        assert np.allclose(blob, np_grad_dict["np_mse_grad_mean"])

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_mseloss(
        of_input: tp.Numpy.Placeholder(shape=input.shape),
        of_target: tp.Numpy.Placeholder(shape=target.shape),
    ) -> Dict[str, tp.Numpy]:
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                shape=input.shape,
                dtype=flow.float32,
                initializer=flow.zeros_initializer(),
                name="x_var",
            )
            x_var = of_input + v

        flow.watch_diff(x_var, assert_prediction_grad)

        mseloss = flow.nn.MSELoss(x_var, of_target, reduction="none", name="of_mseloss")
        mseloss_mean = flow.nn.MSELoss(
            x_var, of_target, reduction="mean", name="of_mseloss_reduce_mean"
        )
        mseloss_sum = flow.nn.MSELoss(
            x_var, of_target, reduction="sum", name="of_mseloss_reduce_sum"
        )

        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(mseloss_mean)

        return {
            "of_mse_loss": mseloss,
            "of_mse_loss_mean": mseloss_mean,
            "of_mse_loss_sum": mseloss_sum,
        }

    of_out_mseloss_dict = oneflow_mseloss(input, target)

    assert np.allclose(
        of_out_mseloss_dict["of_mse_loss"], np_out_mseloss_dict["np_mse_loss"]
    )
    assert np.allclose(
        of_out_mseloss_dict["of_mse_loss_mean"], np_out_mseloss_dict["np_mse_loss_mean"]
    )
    assert np.allclose(
        of_out_mseloss_dict["of_mse_loss_sum"], np_out_mseloss_dict["np_mse_loss_sum"]
    )


def _gen_arg_dict(shape, device_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["input_shape"] = [shape]
    arg_dict["target_shape"] = [shape]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class Testmseloss1n1d(flow.unittest.TestCase):
    def test_mseloss_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 16), device_type="cpu", machine_ids="0:0", device_counts=1
        )

        for arg in GenArgList(arg_dict):
            _compare_mseloss_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_mseloss_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 16, 32), device_type="gpu", machine_ids="0:0", device_counts=1
        )
        for arg in GenArgList(arg_dict):
            _compare_mseloss_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Testmseloss1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_mseloss_gpu_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 16, 16), device_type="gpu", machine_ids="0:0-1", device_counts=2
        )
        for arg in GenArgList(arg_dict):
            _compare_mseloss_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
