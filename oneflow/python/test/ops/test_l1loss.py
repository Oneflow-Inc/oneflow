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


def _compare_l1loss_with_np(
    input_shape, target_shape, device_type, machine_ids, device_counts
):
    input = np.random.random(size=input_shape).astype(np.float32)
    target = np.random.random(size=target_shape).astype(np.float32)

    assert device_type in ["cpu", "gpu"]

    func_config = flow.FunctionConfig()

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))
    func_config.default_logical_view(flow.scope.consistent_view())

    def np_l1loss(np_input, np_target):
        np_l1 = np.abs(np_target - np_input)
        np_l1_mean = np.mean(np_l1)
        np_l1_sum = np.sum(np_l1)

        np_l1_dict = {
            "np_l1_loss": np_l1,
            "np_l1_loss_mean": np_l1_mean,
            "np_l1_loss_sum": np_l1_sum,
        }

        return np_l1_dict

    def np_l1_loss_diff(np_input, np_target):
        # Use numpy to compute diff
        original_shape = np_target.shape
        elemcnt = np_target.size
        prediction = np_input.reshape(-1)
        label = np_target.reshape(-1)
        prediction_grad = np.zeros((elemcnt)).astype(prediction.dtype)

        for i in np.arange(elemcnt):
            diff = prediction[i] - label[i]
            prediction_grad[i] = np.sign(diff)

        grad_mean = prediction_grad.reshape(original_shape) / elemcnt

        # TODO: if you want to get the grad when the reduction = "sum", you can use the follow code
        # grad_sum = prediction_grad.reshape(original_shape)

        grad_dict = {
            "np_grad_mean": grad_mean,
        }

        return grad_dict

    # Use Numpy to compute l1 loss
    np_out_l1loss_dict = np_l1loss(input, target)
    # Use Numpy to compute l1 grad
    np_grad_dict = np_l1_loss_diff(input, target)

    def assert_prediction_grad(blob: tp.Numpy):
        # Evaluate the gradient. Here we only test the reduction type == "mean"
        assert np.allclose(blob, np_grad_dict["np_grad_mean"])

    @flow.global_function(type="train", function_config=func_config)
    def oneflow_l1loss(
        of_input: tp.Numpy.Placeholder(shape=input.shape),
        of_target: tp.Numpy.Placeholder(shape=target.shape),
    ) -> Dict[str, tp.Numpy]:

        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                shape=target.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
                name="v",
            )

            x_var = of_input + v
        # watch the diff
        flow.watch_diff(x_var, assert_prediction_grad)

        l1loss = flow.nn.L1Loss(x_var, of_target, reduction="none", name="of_l1loss")
        l1loss_mean = flow.nn.L1Loss(
            x_var, of_target, reduction="mean", name="of_l1loss_mean"
        )
        l1loss_sum = flow.nn.L1Loss(
            x_var, of_target, reduction="sum", name="of_l1loss_sum"
        )

        with flow.scope.placement(device_type, "0:0"):
            # We only test reduction="mean" diff
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(l1loss_mean)

        return {
            "of_l1_loss": l1loss,
            "of_l1_loss_mean": l1loss_mean,
            "of_l1_loss_sum": l1loss_sum,
        }

    of_out_l1loss_dict = oneflow_l1loss(input, target)

    assert np.allclose(
        of_out_l1loss_dict["of_l1_loss"], np_out_l1loss_dict["np_l1_loss"]
    )
    assert np.allclose(
        of_out_l1loss_dict["of_l1_loss_mean"][0], np_out_l1loss_dict["np_l1_loss_mean"]
    )
    assert np.allclose(
        of_out_l1loss_dict["of_l1_loss_sum"][0], np_out_l1loss_dict["np_l1_loss_sum"]
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
class Testl1loss1n1d(flow.unittest.TestCase):
    def test_l1loss_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(16, 3), device_type="cpu", machine_ids="0:0", device_counts=1
        )
        for arg in GenArgList(arg_dict):
            _compare_l1loss_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_l1loss_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 16, 32), device_type="gpu", machine_ids="0:0", device_counts=1
        )
        for arg in GenArgList(arg_dict):
            _compare_l1loss_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Testl1loss1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_l1loss_gpu_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 32, 16), device_type="gpu", machine_ids="0:0-1", device_counts=2
        )
        for arg in GenArgList(arg_dict):
            _compare_l1loss_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
