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


def _compare_margin_ranking_loss_with_np(
    input1_shape,
    input2_shape,
    target_shape,
    margin,
    device_type,
    machine_ids,
    device_counts,
):
    input1 = np.random.random(size=input1_shape).astype(np.float32)
    input2 = np.random.random(size=input2_shape).astype(np.float32)
    target = np.random.random(size=target_shape).astype(np.float32)

    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))
    func_config.default_logical_view(flow.scope.consistent_view())

    def np_margin_ranking_loss(np_input1, np_input2, np_target, np_margin):
        np_target = np.broadcast_to(np_target, shape=(np_input1.shape))
        np_margin_loss = np.maximum(0, -(np_input1 - np_input2) * np_target + np_margin)
        np_margin_loss_mean = np.mean(np_margin_loss)
        np_margin_loss_sum = np.sum(np_margin_loss)

        return {
            "np_margin_ranking_loss": np_margin_loss,
            "np_margin_ranking_loss_mean": np_margin_loss_mean,
            "np_margin_ranking_loss_sum": np_margin_loss_sum,
        }

    np_out_marginloss_dict = np_margin_ranking_loss(input1, input2, target, margin)

    def np_margin_ranking_diff(np_out, np_target):
        # Use numpy to compute diff
        # Here we only test the input_1 gradient
        # If loss > 0, the grad is: -target, else the grad is 0
        _elem_cnt = np_out.size

        if np_out.shape != np_target.shape:
            # Do Broadcast
            np_target = np.broadcast_to(np_target, shape=(np_out.shape))

        # If out > 0, the element = 1, else set to 0
        _clip_zero_index = np.where(np_out > 0, 1, 0)

        _np_grad = -np_target

        return {"np_margin_ranking_grad_mean": _np_grad * _clip_zero_index / _elem_cnt}

    np_grad_dict = np_margin_ranking_diff(
        np_out_marginloss_dict["np_margin_ranking_loss"], target
    )

    def assert_prediction_grad(blob: tp.Numpy):
        # Evaluate the gradient
        # Here we only test the input_1 gradient
        # If loss > 0, the grad is: -target, else the grad is 0
        assert np.allclose(blob, np_grad_dict["np_margin_ranking_grad_mean"])

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_marginloss(
        of_input1: tp.Numpy.Placeholder(shape=input1.shape),
        of_input2: tp.Numpy.Placeholder(shape=input2.shape),
        of_target: tp.Numpy.Placeholder(shape=target.shape),
    ) -> Dict[str, tp.Numpy]:
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                shape=input1.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
                name="x_var",
            )
            x_var = of_input1 + v

        flow.watch_diff(x_var, assert_prediction_grad)

        marginloss = flow.nn.MarginRankingLoss(
            of_input1,
            of_input2,
            of_target,
            margin=margin,
            reduction="none",
            name="of_marginloss",
        )
        marginloss_mean = flow.nn.MarginRankingLoss(
            x_var,
            of_input2,
            of_target,
            margin=margin,
            reduction="mean",
            name="of_marginloss_reduce_mean",
        )
        marginloss_sum = flow.nn.MarginRankingLoss(
            of_input1,
            of_input2,
            of_target,
            margin=margin,
            reduction="sum",
            name="of_marginloss_reduce_sum",
        )

        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(marginloss_mean)

        return {
            "of_margin_ranking_loss": marginloss,
            "of_margin_ranking_loss_mean": marginloss_mean,
            "of_margin_ranking_loss_sum": marginloss_sum,
        }

    of_out_marginloss_dict = oneflow_marginloss(input1, input2, target)

    assert np.allclose(
        of_out_marginloss_dict["of_margin_ranking_loss"],
        np_out_marginloss_dict["np_margin_ranking_loss"],
    )
    assert np.allclose(
        of_out_marginloss_dict["of_margin_ranking_loss_mean"],
        np_out_marginloss_dict["np_margin_ranking_loss_mean"],
    )
    assert np.allclose(
        of_out_marginloss_dict["of_margin_ranking_loss_sum"],
        np_out_marginloss_dict["np_margin_ranking_loss_sum"],
    )


def _gen_arg_dict(shape, target_shape, margin, device_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["input1_shape"] = [shape]
    arg_dict["input2_shape"] = [shape]
    arg_dict["target_shape"] = [target_shape]
    arg_dict["margin"] = [margin]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class Testmarginloss1n1d(flow.unittest.TestCase):
    def test_margin_ranking_loss_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 5),
            target_shape=(3, 1),
            margin=0.3,
            device_type="cpu",
            machine_ids="0:0",
            device_counts=1,
        )

        for arg in GenArgList(arg_dict):
            _compare_margin_ranking_loss_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_margin_ranking_loss_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(4, 5),
            target_shape=(4, 1),
            margin=0.3,
            device_type="gpu",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_margin_ranking_loss_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Testmarginloss1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_margin_ranking_loss_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 3),
            target_shape=(3, 1),
            margin=0.3,
            device_type="gpu",
            machine_ids="0:0-1",
            device_counts=2,
        )
        for arg in GenArgList(arg_dict):
            _compare_margin_ranking_loss_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
