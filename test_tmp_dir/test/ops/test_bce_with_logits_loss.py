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


def _compare_bce_with_logits_loss_np(
    input_shape,
    target_shape,
    weight_shape,
    pos_weight_shape,
    device_type,
    machine_ids,
    device_counts,
):
    # random samples from a uniform distribution over [0, 1)
    input = (
        np.random.random(size=input_shape).astype(np.float32) - 0.5
    )  # change the distribution to [-0.5, 0.5)
    target = np.random.random(size=target_shape).astype(np.float32) - 0.5
    pos_weight = np.random.random(size=pos_weight_shape).astype(np.float32)
    weight = np.random.random(size=weight_shape).astype(np.float32)

    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))

    def np_bce_with_logits_loss(np_input, np_target, np_weight, np_pos_weight):
        max_val = np.clip(-np_input, a_min=0, a_max=1e6)
        if pos_weight.any():
            log_weight = ((np_pos_weight - 1) * np_target) + 1
            loss = (1 - np_target) * np_input
            loss_1 = np.log(np.exp(-max_val) + np.exp(-np_input - max_val)) + max_val
            loss += log_weight * loss_1
        else:
            loss = (1 - np_target) * np_input
            loss += max_val
            loss += np.log(np.exp(-max_val) + np.exp(-np_input - max_val))

        np_bce = loss * np_weight

        np_bce_mean = np.mean(np_bce)
        np_bce_sum = np.sum(np_bce)

        return {
            "np_bce_with_logits_loss": np_bce,
            "np_bce_with_logits_loss_mean": np_bce_mean,
            "np_bce_with_logits_loss_sum": np_bce_sum,
        }

    def np_bce_with_logits_loss_diff(np_input, np_target, np_weight, np_pos_weight):
        # Use numpy to compute diff
        elemcnt = np_target.size

        np_bce_with_logits_grad_mean = -(np_weight / elemcnt) * (
            (np_target - 1)
            + ((1 - np_pos_weight) * np_target - 1)
            * (-np.exp(-np_input) / (1 + np.exp(-np_input)))
        )

        return {
            "np_bce_with_logits_grad_mean": np_bce_with_logits_grad_mean,
        }

    np_out_bceloss_dict = np_bce_with_logits_loss(input, target, weight, pos_weight)

    # Compute diff
    np_grad_dict = np_bce_with_logits_loss_diff(input, target, weight, pos_weight)

    def assert_prediction_grad(blob: tp.Numpy):
        # Evaluate the gradient
        assert np.allclose(blob, np_grad_dict["np_bce_with_logits_grad_mean"])

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_bce_with_logits_loss(
        of_input: tp.Numpy.Placeholder(shape=input.shape),
        of_target: tp.Numpy.Placeholder(shape=target.shape),
        of_weight: tp.Numpy.Placeholder(shape=weight.shape),
        of_pos_weight: tp.Numpy.Placeholder(shape=pos_weight.shape),
    ) -> Dict[str, tp.Numpy]:
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                shape=input.shape,
                dtype=flow.float32,
                initializer=flow.zeros_initializer(),
                name="v",
            )

            x_var = of_input + v

        flow.watch_diff(x_var, assert_prediction_grad)

        bceloss = flow.nn.BCEWithLogitsLoss(
            x_var,
            of_target,
            of_weight,
            of_pos_weight,
            reduction="none",
            name="of_mseloss",
        )
        bceloss_mean = flow.nn.BCEWithLogitsLoss(
            x_var,
            of_target,
            of_weight,
            of_pos_weight,
            reduction="mean",
            name="of_mseloss_reduce_mean",
        )
        bceloss_sum = flow.nn.BCEWithLogitsLoss(
            x_var,
            of_target,
            of_weight,
            of_pos_weight,
            reduction="sum",
            name="of_mseloss_reduce_sum",
        )
        # Because our gradient is use "mean" mode to compute
        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(bceloss_mean)

        return {
            "of_bce_with_logits_loss": bceloss,
            "of_bce_with_logits_loss_mean": bceloss_mean,
            "of_bce_with_logits_loss_sum": bceloss_sum,
        }

    of_out_bceloss_dict = oneflow_bce_with_logits_loss(
        input, target, weight, pos_weight
    )

    assert np.allclose(
        of_out_bceloss_dict["of_bce_with_logits_loss"],
        np_out_bceloss_dict["np_bce_with_logits_loss"],
    )
    assert np.allclose(
        of_out_bceloss_dict["of_bce_with_logits_loss_mean"][0],
        np_out_bceloss_dict["np_bce_with_logits_loss_mean"],
    )
    assert np.allclose(
        of_out_bceloss_dict["of_bce_with_logits_loss_sum"][0],
        np_out_bceloss_dict["np_bce_with_logits_loss_sum"],
    )


def _gen_arg_dict(shape, device_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["input_shape"] = [shape]
    arg_dict["target_shape"] = [shape]
    arg_dict["weight_shape"] = [shape]
    # The pos weight shape must be equal to Classes
    # so I choose the last channel
    arg_dict["pos_weight_shape"] = [shape[-1]]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class TestBCEWithLogitsLoss1n1d(flow.unittest.TestCase):
    def test_bce_with_logits_loss_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 3), device_type="cpu", machine_ids="0:0", device_counts=1
        )

        for arg in GenArgList(arg_dict):
            _compare_bce_with_logits_loss_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_bce_with_logits_loss_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(4, 16), device_type="gpu", machine_ids="0:0", device_counts=1
        )
        for arg in GenArgList(arg_dict):
            _compare_bce_with_logits_loss_np(*arg)


@flow.unittest.skip_unless_1n2d()
class TestBCEWithLogits1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_bce_with_logits_gpu_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            shape=(4, 8), device_type="gpu", machine_ids="0:0-1", device_counts=2
        )
        for arg in GenArgList(arg_dict):
            _compare_bce_with_logits_loss_np(*arg)


if __name__ == "__main__":
    unittest.main()
