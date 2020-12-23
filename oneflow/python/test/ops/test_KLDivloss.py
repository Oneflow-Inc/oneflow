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


def _compare_kldivloss_with_np(
    input_shape, target_shape, log_target, device_type, machine_ids, device_counts,
):
    input = np.random.random(size=input_shape).astype(np.float32)
    target = np.random.random(size=target_shape).astype(np.float32)

    log_target = log_target[0]
    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))
    func_config.default_logical_view(flow.scope.consistent_view())

    def np_kldivloss(np_input, np_target, np_log_target):
        if log_target:
            np_kl_div_loss = np.exp(np_target) * (np_target - np_input)
        else:
            np_kl_div_out_loss = target * (np.log(target) - np_input)
            np_zeros = np.zeros_like(np_kl_div_out_loss, dtype=np.float32)
            # when target < 0, we set to `0`, when target > 0, we set to `1`.
            # set the element in _kl_div_loss as `0` to avoid `nan` value.
            np_kl_div_loss = np.where(target > 0, np_kl_div_out_loss, np_zeros)

        return {
            "np_kldivloss": np_kl_div_loss,
            "np_kldivloss_mean": np.mean(np_kl_div_loss),
            "np_kldivloss_sum": np.sum(np_kl_div_loss),
        }

    np_out_kldivloss_dict = np_kldivloss(input, target, log_target)

    def np_kldivloss_diff(input, target, np_log_target):
        elem_cnt = input.size
        if np_log_target:
            _np_diff = -np.exp(target)
        else:
            _np_diff = -target
            # Because when np_log_target == False, the loss will be set to zero when target < 0
            _zero_index = np.where(target > 0, 1, 0)
            _np_diff = _np_diff * _zero_index

        return {
            "np_kldivloss_grad": _np_diff,
            "np_kldivloss_grad_mean": _np_diff / elem_cnt,
        }

    np_grad_dict = np_kldivloss_diff(input, target, log_target)

    def assert_prediction_grad(blob: tp.Numpy):
        # validate the correstness of gradient
        assert np.allclose(blob, np_grad_dict["np_kldivloss_grad_mean"], rtol=1e-4)

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_kldivloss(
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
            of_input = of_input + v

        flow.watch_diff(of_input, assert_prediction_grad)

        of_kldivloss = flow.nn.KLDivLoss(
            of_input,
            of_target,
            log_target=log_target,
            reduction="none",
            name="kldivloss",
        )
        of_kldivloss_mean = flow.nn.KLDivLoss(
            of_input,
            of_target,
            log_target=log_target,
            reduction="mean",
            name="kldivloss_mean",
        )
        of_kldivloss_sum = flow.nn.KLDivLoss(
            of_input,
            of_target,
            log_target=log_target,
            reduction="sum",
            name="kldivloss_sum",
        )

        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(of_kldivloss_mean)

        return {
            "of_kldivloss": of_kldivloss,
            "of_kldivloss_mean": of_kldivloss_mean,
            "of_kldivloss_sum": of_kldivloss_sum,
        }

    of_out_kldivloss_dict = oneflow_kldivloss(input, target)

    assert np.allclose(
        of_out_kldivloss_dict["of_kldivloss"], np_out_kldivloss_dict["np_kldivloss"],
    )

    assert np.allclose(
        of_out_kldivloss_dict["of_kldivloss_mean"],
        np_out_kldivloss_dict["np_kldivloss_mean"],
    )
    assert np.allclose(
        of_out_kldivloss_dict["of_kldivloss_sum"],
        np_out_kldivloss_dict["np_kldivloss_sum"],
    )


def _gen_arg_dict(shape, log_target, device_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["input_shape"] = [shape]
    arg_dict["target_shape"] = [shape]
    arg_dict["log_target"] = [log_target]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class Test_KLDivLoss_1n1d(flow.unittest.TestCase):
    def test_kldivloss_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 3),
            log_target=[True],
            device_type="cpu",
            machine_ids="0:0",
            device_counts=1,
        )

        for arg in GenArgList(arg_dict):
            _compare_kldivloss_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_kldivloss_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(4, 4),
            log_target=[False],
            device_type="gpu",
            machine_ids="0:0",
            device_counts=1,
        )

        for arg in GenArgList(arg_dict):
            _compare_kldivloss_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Test_KLDivLoss_1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_kldivloss_gpu_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            shape=(4, 4),
            log_target=[True],
            device_type="gpu",
            machine_ids="0:0-1",
            device_counts=2,
        )

        for arg in GenArgList(arg_dict):
            _compare_kldivloss_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
