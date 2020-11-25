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


def _compare_bceloss_with_np(
    input_shape, target_shape, weight_shape, device_type, machine_ids, device_counts
):
    input = np.random.random(size=input_shape).astype(np.float32)
    target = np.random.random(size=target_shape).astype(np.float32)
    weight = np.random.random(size=weight_shape).astype(np.float32)

    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    flow.env.init()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()

    def _np_sigmoid_fn(x):
        # Compute sigmoid function
        return 1 / (1 + np.exp(-x))

    def np_bceloss(np_input, np_target, np_weight):
        np_sigmoid_input = _np_sigmoid_fn(np_input)
        np_bce = -np_weight * (
            (
                np_target * np.log(np_sigmoid_input)
                + (1 - np_target) * (np.log(1 - np_sigmoid_input))
            )
        )

        np_bce_mean = np.mean(np_bce)
        np_bce_sum = np.sum(np_bce)

        return {
            "np_bce_loss": np_bce,
            "np_bce_loss_mean": np_bce_mean,
            "np_bce_loss_sum": np_bce_sum,
        }

    def np_bce_loss_diff(np_input, np_target, np_weight):
        # Use numpy to compute diff
        elemcnt = np_target.size

        # TODO: If you want to get the grad when the reduction = "sum", you can use the follow code

        # np_bce_grad_sum = -(np_weight) * (
        # np_target - (np.exp(np_input) / (1 + np.exp(np_input)))
        # )

        np_bce_grad_mean = -(np_weight / elemcnt) * (
            np_target - (np.exp(np_input) / (1 + np.exp(np_input)))
        )

        return {
            "np_bce_grad_mean": np_bce_grad_mean,
        }

    np_out_bceloss_dict = np_bceloss(input, target, weight)

    # Compute diff
    np_grad_dict = np_bce_loss_diff(input, target, weight)

    def assert_prediction_grad(blob: tp.Numpy):
        # Evaluate the gradient
        assert np.allclose(blob, np_grad_dict["np_bce_grad_mean"])

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_bceloss(
        of_input: tp.Numpy.Placeholder(shape=input.shape),
        of_target: tp.Numpy.Placeholder(shape=target.shape),
        of_weight: tp.Numpy.Placeholder(shape=weight.shape),
    ) -> Dict[str, tp.Numpy]:
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                shape=target.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(1),
                name="v",
            )

            x_var = of_input + v

        flow.watch_diff(x_var, assert_prediction_grad)

        bceloss = flow.nn.BCELoss(
            x_var, of_target, of_weight, reduction="none", name="of_mseloss"
        )
        bceloss_mean = flow.nn.BCELoss(
            x_var,
            of_target,
            of_weight,
            reduction="mean",
            name="of_mseloss_reduce_mean",
        )
        bceloss_sum = flow.nn.BCELoss(
            x_var, of_target, of_weight, reduction="sum", name="of_mseloss_reduce_sum",
        )
        # Because our gradient is use "mean" mode to compute
        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(bceloss_mean)

        return {
            "of_bce_loss": bceloss,
            "of_bce_loss_mean": bceloss_mean,
            "of_bce_loss_sum": bceloss_sum,
        }

    of_out_bceloss_dict = oneflow_bceloss(input, target, weight)

    assert np.allclose(
        of_out_bceloss_dict["of_bce_loss"], np_out_bceloss_dict["np_bce_loss"]
    )
    assert np.allclose(
        of_out_bceloss_dict["of_bce_loss_mean"][0],
        np_out_bceloss_dict["np_bce_loss_mean"],
    )
    assert np.allclose(
        of_out_bceloss_dict["of_bce_loss_sum"][0],
        np_out_bceloss_dict["np_bce_loss_sum"],
    )


def _gen_arg_dict(shape, device_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["input_shape"] = [shape]
    arg_dict["target_shape"] = [shape]
    arg_dict["weight_shape"] = [shape]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class Testbceloss1n1d(flow.unittest.TestCase):
    def test_bceloss_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 16), device_type="cpu", machine_ids="0:0", device_counts=1
        )

        for arg in GenArgList(arg_dict):
            _compare_bceloss_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_bceloss_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 16, 32), device_type="gpu", machine_ids="0:0", device_counts=1
        )
        for arg in GenArgList(arg_dict):
            _compare_bceloss_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Testbceloss1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_bceloss_gpu_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 16, 16), device_type="gpu", machine_ids="0:0-1", device_counts=2
        )
        for arg in GenArgList(arg_dict):
            _compare_bceloss_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
