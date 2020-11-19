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

    assert device_type in ["cpu", "gpu"]

    flow.clear_default_session()
    if device_type == "cpu":
        flow.config.cpu_device_num(device_counts)
    else:
        flow.config.gpu_device_num(device_counts)

    func_config = flow.FunctionConfig()

    def np_triplet_margin_loss(np_anchor, np_pos, np_neg, np_margin, np_p, swap):
        np_d_1_norm = np.power(np.abs((np_anchor-np_pos+1e-6)), np_p)
        np_d_2_norm = np.power(np.abs((np_anchor-np_neg+1e-6)), np_p)

        np_d_1 = np.power(np.sum(np_d_1_norm, axis=-1), 1./p)
        np_d_2 = np.power(np.sum(np_d_2_norm, axis=-1), 1./p)

        if swap:
            np_dist_swap = np.sum(np.square(np_pos - np_neg+1e-6), axis=-1)
            np_d_2 = np.minimum(np_d_2, np_dist_swap)
        
        np_triplet_margin_loss = (margin + np_d_1 - np_d_2)
        np_triplet_margin_loss_mean = np.mean(np_triplet_margin_loss)
        np_triplet_margin_loss_sum = np.sum(np_triplet_margin_loss)

        return {
            "np_triplet_margin_loss": np_triplet_margin_loss,
            "np_triplet_margin_loss_mean": np_triplet_margin_loss_mean,
            "np_triplet_margin_loss_sum": np_triplet_margin_loss_sum,
        }

    np_out_tripletloss_dict = np_triplet_margin_loss(anchor, pos, neg, margin, p, swap)

    # def np_margin_ranking_diff(np_out, np_target):
    #     # Use numpy to compute diff
    #     # Here we only test the input_1 gradient
    #     # If loss > 0, the grad is: -target, else the grad is 0
    #     elem_cnt = np_out.size
    #     row, col = np_out.shape
    #     np_diff = np.zeros(shape=np_out.shape)
    #     # TODO: Optimize the backward logic
    #     for i in range(row):
    #         for j in range(col):
    #             if np_out[i][j] > 0:
    #                 np_diff[i][j] = -np_target[i]

    #     return {
    #         "np_margin_ranking_grad_mean": np_diff / elem_cnt,
    #     }

    # np_grad_dict = np_margin_ranking_diff(
    #     np_out_marginloss_dict["np_margin_ranking_loss"], target
    # )

    # def assert_prediction_grad(blob: tp.Numpy):
    #     # Evaluate the gradient
    #     # Here we only test the input_1 gradient
    #     # If loss > 0, the grad is: -target, else the grad is 0
    #     assert np.allclose(blob, np_grad_dict["np_margin_ranking_grad_mean"])

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

        # TODO: add watch diff
        # flow.watch_diff(x_var, assert_prediction_grad)

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

    print("Of loss is: ", of_out_tripletloss_dict["of_triplet_margin_loss_mean"])
    print("np loss is: ", np_out_tripletloss_dict["np_triplet_margin_loss_mean"])

    assert np.allclose(
        of_out_tripletloss_dict["of_triplet_margin_loss_mean"],
        np_out_tripletloss_dict["np_triplet_margin_loss_mean"],
    )
    assert np.allclose(
        of_out_tripletloss_dict["of_triplet_margin_loss_sum"],
        np_out_tripletloss_dict["np_triplet_margin_loss_sum"],
    )


def _gen_arg_dict(shape, margin, p, swap, device_type, machine_ids, device_counts):
    # Generate a dict to pass parameter to test case
    arg_dict = OrderedDict()
    arg_dict["anchor_shape"] = [shape]
    arg_dict["pos_shape"] = [shape]
    arg_dict["neg_shape"] = [shape]
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
            shape=(3, 5),
            margin=1, 
            p=2.0, 
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