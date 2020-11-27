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


def _compare_triplet_margin_loss_with_np(
    anchor_shape,
    pos_shape,
    neg_shape,
    eps,
    margin,
    p,
    swap,
    device_type,
    machine_ids,
    device_counts,
):
    anchor = np.random.random(size=anchor_shape).astype(np.float32)
    pos = np.random.random(size=pos_shape).astype(np.float32)
    neg = np.random.random(size=neg_shape).astype(np.float32)
    eps = eps

    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()
    func_config.default_placement_scope(flow.scope.placement(device_type, machine_ids))
    func_config.default_logical_view(flow.scope.consistent_view())

    def np_triplet_margin_loss(np_anchor, np_pos, np_neg, eps, np_margin, np_p, swap):
        np_d_1_norm = np.power(np.abs((np_anchor - np_pos + eps)), np_p)
        np_d_2_norm = np.power(np.abs((np_anchor - np_neg + eps)), np_p)

        np_d_1 = np.power(np.sum(np_d_1_norm, axis=-1), 1.0 / np_p)
        np_d_2 = np.power(np.sum(np_d_2_norm, axis=-1), 1.0 / np_p)

        if swap:
            np_dist_swap = np.power(np.abs((np_pos - np_neg + eps)), np_p)
            np_dist_swap = np.power(np.sum(np_dist_swap, axis=-1), 1.0 / np_p)
            np_d_2 = np.minimum(np_d_2, np_dist_swap)

        np_triplet_margin_loss = np.maximum((np_margin + np_d_1 - np_d_2), 0)
        np_triplet_margin_loss_mean = np.mean(np_triplet_margin_loss)
        np_triplet_margin_loss_sum = np.sum(np_triplet_margin_loss)

        return {
            "np_triplet_margin_loss": np_triplet_margin_loss,
            "np_triplet_margin_loss_mean": np_triplet_margin_loss_mean,
            "np_triplet_margin_loss_sum": np_triplet_margin_loss_sum,
        }

    np_out_tripletloss_dict = np_triplet_margin_loss(
        anchor, pos, neg, eps, margin, p, swap
    )

    def np_triplet_loss_diff(anchor, pos, neg, margin, p):
        def _compute_distance(x1, x2, x3):
            d_1_norm = np.power(np.abs((x1 - x2 + 1e-6)), p)
            d_2_norm = np.power(np.abs((x1 - x3 + 1e-6)), p)
            d_1 = np.power(np.sum(d_1_norm, axis=-1), 1.0 / p)
            d_2 = np.power(np.sum(d_2_norm, axis=-1), 1.0 / p)

            return d_1 - d_2 + margin

        def _compute_per_diff(x1, x2, p, eps=1e-6):
            # Add epsilon to avoid divided by zero
            _abs_index = np.where(x1 - x2 > 0, 1, -1)
            # When element == 0, its grad = 0
            _abs_index_support = np.where(x1 - x2 == 0, 1, 0)
            _abs_grad = _abs_index + _abs_index_support

            _abs_val = np.abs(x1 - x2 + eps)
            _power_abs_val = np.power(_abs_val, p)
            _sum_val = np.sum(_power_abs_val, axis=1, keepdims=True)

            # Add epsilon to avoid divided by zero
            _sqrt_sum_val = np.power(_sum_val + eps, 1.0 / p - 1)

            _power_val = np.power(_abs_val, p - 1)

            _grad = np.multiply(_sqrt_sum_val, _power_val)
            # Multiply the abs grad
            _grad *= _abs_grad
            return _grad / x1.shape[0]

        d = _compute_distance(anchor, pos, neg)
        # Because We use max(x, 0), the value less than 0, the corresponding grad is 0
        # So Here we compute the index that its grad need to be place to 0
        zero_index = np.where(d < -1e-6)

        anchor_grad_1 = _compute_per_diff(anchor, pos, p)
        anchor_grad_2 = _compute_per_diff(anchor, neg, p)

        total_grad = anchor_grad_1 - anchor_grad_2

        for i in zero_index:
            total_grad[i] = 0

        grad_dict = {
            "np_triplet_loss_grad_mean": total_grad,
        }

        return grad_dict

    np_grad_dict = np_triplet_loss_diff(anchor, pos, neg, margin, p)

    def assert_prediction_grad(blob: tp.Numpy):
        # Evaluate the gradient
        assert np.allclose(blob, np_grad_dict["np_triplet_loss_grad_mean"], rtol=1e-3)

    @flow.global_function(
        type="train", function_config=func_config,
    )
    def oneflow_marginloss(
        of_anchor: tp.Numpy.Placeholder(shape=anchor.shape),
        of_pos: tp.Numpy.Placeholder(shape=pos.shape),
        of_neg: tp.Numpy.Placeholder(shape=neg.shape),
    ) -> Dict[str, tp.Numpy]:
        with flow.scope.placement(device_type, "0:0"):
            v = flow.get_variable(
                shape=anchor.shape,
                dtype=flow.float32,
                initializer=flow.constant_initializer(0),
                name="x_var",
            )
            x_anchor = of_anchor + v

        flow.watch_diff(x_anchor, assert_prediction_grad)

        triplet_marginloss = flow.nn.TripletMarginLoss(
            x_anchor,
            of_pos,
            of_neg,
            margin=margin,
            p=p,
            swap=swap,
            reduction="none",
            name="of_tripletmarginloss",
        )
        triplet_marginloss_mean = flow.nn.TripletMarginLoss(
            x_anchor,
            of_pos,
            of_neg,
            margin=margin,
            p=p,
            swap=swap,
            reduction="mean",
            name="of_tripletmarginloss_mean",
        )
        triplet_marginloss_sum = flow.nn.TripletMarginLoss(
            x_anchor,
            of_pos,
            of_neg,
            margin=margin,
            p=p,
            swap=swap,
            reduction="sum",
            name="of_tripletmarginloss_sum",
        )

        with flow.scope.placement(device_type, "0:0"):
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1e-3]), momentum=0
            ).minimize(triplet_marginloss_mean)

        return {
            "of_triplet_margin_loss": triplet_marginloss,
            "of_triplet_margin_loss_mean": triplet_marginloss_mean,
            "of_triplet_margin_loss_sum": triplet_marginloss_sum,
        }

    of_out_tripletloss_dict = oneflow_marginloss(anchor, pos, neg)

    assert np.allclose(
        of_out_tripletloss_dict["of_triplet_margin_loss"],
        np_out_tripletloss_dict["np_triplet_margin_loss"],
    )

    assert np.allclose(
        of_out_tripletloss_dict["of_triplet_margin_loss_mean"],
        np_out_tripletloss_dict["np_triplet_margin_loss_mean"],
    )
    assert np.allclose(
        of_out_tripletloss_dict["of_triplet_margin_loss_sum"],
        np_out_tripletloss_dict["np_triplet_margin_loss_sum"],
    )


def _gen_arg_dict(shape, eps, margin, p, swap, device_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["anchor_shape"] = [shape]
    arg_dict["pos_shape"] = [shape]
    arg_dict["neg_shape"] = [shape]
    arg_dict["eps"] = [eps]
    arg_dict["margin"] = [margin]
    arg_dict["p"] = [p]
    arg_dict["swap"] = [swap]
    arg_dict["device_type"] = [device_type]
    arg_dict["machine_ids"] = [machine_ids]
    arg_dict["device_counts"] = [device_counts]
    return arg_dict


@flow.unittest.skip_unless_1n1d()
class Test_triplet_loss_1n1d(flow.unittest.TestCase):
    def test_triplet_margin_loss_cpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 3),
            eps=1e-6,
            margin=1,
            p=1.5,
            swap=False,
            device_type="cpu",
            machine_ids="0:0",
            device_counts=1,
        )

        for arg in GenArgList(arg_dict):
            _compare_triplet_margin_loss_with_np(*arg)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_margin_ranking_loss_gpu(test_case):
        arg_dict = _gen_arg_dict(
            shape=(3, 6),
            eps=1e-6,
            margin=1,
            p=2.0,
            swap=False,
            device_type="gpu",
            machine_ids="0:0",
            device_counts=1,
        )
        for arg in GenArgList(arg_dict):
            _compare_triplet_margin_loss_with_np(*arg)


@flow.unittest.skip_unless_1n2d()
class Testmarginloss1n2d(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_margin_ranking_loss_1n2d(test_case):
        arg_dict = _gen_arg_dict(
            shape=(6, 6),
            eps=1e-6,
            margin=1,
            p=2.0,
            swap=False,
            device_type="gpu",
            machine_ids="0:0-1",
            device_counts=2,
        )
        for arg in GenArgList(arg_dict):
            _compare_triplet_margin_loss_with_np(*arg)


if __name__ == "__main__":
    unittest.main()
