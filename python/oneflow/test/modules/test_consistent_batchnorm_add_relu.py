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
from collections import OrderedDict
import os

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


def _test_bn_add_relu(test_case, batch, channel, height, width, placement, sbp):
    weight = random_tensor(1, channel).oneflow
    bias = random_tensor(1, channel).oneflow
    x = random_tensor(4, batch, channel, height, width).oneflow
    addend = random_tensor(4, batch, channel, height, width).oneflow

    track_running_stats = random_bool().value()

    params_sbp = [flow.sbp.broadcast for _ in range(len(sbp))]
    fused_bn = flow.nn.FusedBatchNorm2d(
        channel, track_running_stats=track_running_stats
    ).to_global(placement=placement, sbp=params_sbp)
    fused_bn.weight = flow.nn.Parameter(
        weight.to_global(placement=placement, sbp=params_sbp)
    )
    fused_bn.bias = flow.nn.Parameter(
        bias.to_global(placement=placement, sbp=params_sbp)
    )

    fused_x = x.to_global(placement=placement, sbp=sbp)
    fused_x.retain_grad()
    fused_addend = addend.to_global(placement=placement, sbp=sbp)
    fused_addend.retain_grad()
    fused_out = fused_bn(fused_x, fused_addend)
    fused_out.sum().backward()

    device = placement.type
    origin_bn = flow.nn.BatchNorm2d(
        channel, track_running_stats=track_running_stats
    ).to(device)
    origin_bn.weight = flow.nn.Parameter(weight.to_local().to(device))
    origin_bn.bias = flow.nn.Parameter(bias.to_local().to(device))

    origin_x = x.to_local().to(device)
    origin_x.retain_grad()
    origin_addend = addend.to_local().to(device)
    origin_addend.retain_grad()
    origin_out = flow.nn.functional.relu(origin_bn(origin_x) + origin_addend)
    origin_out.sum().backward()

    # test output.
    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )
    # test input grad.
    test_case.assertTrue(
        np.allclose(fused_x.grad.numpy(), origin_x.grad.numpy(), atol=1e-4, rtol=1e-4,)
    )
    test_case.assertTrue(
        np.allclose(
            fused_addend.grad.numpy(), origin_addend.grad.numpy(), atol=1e-4, rtol=1e-4,
        )
    )
    # test weight and bias grad.
    test_case.assertTrue(
        np.allclose(
            fused_bn.weight.grad.numpy(),
            origin_bn.weight.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_bn.bias.grad.numpy(),
            origin_bn.bias.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    # test running mean and running variance.
    if track_running_stats:
        test_case.assertTrue(
            np.allclose(
                fused_bn.running_mean.numpy(),
                origin_bn.running_mean.numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )
        test_case.assertTrue(
            np.allclose(
                fused_bn.running_var.numpy(),
                origin_bn.running_var.numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )
    else:
        test_case.assertIsNone(fused_bn.running_mean)
        test_case.assertIsNone(origin_bn.running_mean)
        test_case.assertIsNone(fused_bn.running_var)
        test_case.assertIsNone(origin_bn.running_var)


def _test_bn_relu(test_case, batch, channel, height, width, placement, sbp):
    weight = random_tensor(1, channel).oneflow
    bias = random_tensor(1, channel).oneflow
    x = random_tensor(4, batch, channel, height, width).oneflow

    track_running_stats = random_bool().value()

    params_sbp = [flow.sbp.broadcast for _ in range(len(sbp))]
    fused_bn = flow.nn.FusedBatchNorm2d(
        channel, track_running_stats=track_running_stats
    ).to_global(placement=placement, sbp=params_sbp)
    fused_bn.weight = flow.nn.Parameter(
        weight.to_global(placement=placement, sbp=params_sbp)
    )
    fused_bn.bias = flow.nn.Parameter(
        bias.to_global(placement=placement, sbp=params_sbp)
    )

    fused_x = x.to_global(placement=placement, sbp=sbp)
    fused_x.retain_grad()
    fused_out = fused_bn(fused_x, None)
    fused_out.sum().backward()

    device = placement.type
    origin_bn = flow.nn.BatchNorm2d(
        channel, track_running_stats=track_running_stats
    ).to(device)
    origin_bn.weight = flow.nn.Parameter(weight.to_local().to(device))
    origin_bn.bias = flow.nn.Parameter(bias.to_local().to(device))

    origin_x = x.to_local().to(device)
    origin_x.retain_grad()
    origin_out = flow.nn.functional.relu(origin_bn(origin_x))
    origin_out.sum().backward()

    # test output.
    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )
    # test input grad.
    test_case.assertTrue(
        np.allclose(fused_x.grad.numpy(), origin_x.grad.numpy(), atol=1e-4, rtol=1e-4,)
    )
    # test weight and bias grad.
    test_case.assertTrue(
        np.allclose(
            fused_bn.weight.grad.numpy(),
            origin_bn.weight.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_bn.bias.grad.numpy(),
            origin_bn.bias.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    # test running mean and running variance.
    if track_running_stats:
        test_case.assertTrue(
            np.allclose(
                fused_bn.running_mean.numpy(),
                origin_bn.running_mean.numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )
        test_case.assertTrue(
            np.allclose(
                fused_bn.running_var.numpy(),
                origin_bn.running_var.numpy(),
                atol=1e-4,
                rtol=1e-4,
            )
        )
    else:
        test_case.assertIsNone(fused_bn.running_mean)
        test_case.assertIsNone(origin_bn.running_mean)
        test_case.assertIsNone(fused_bn.running_var)
        test_case.assertIsNone(origin_bn.running_var)


def _test_bn_add_relu_eval(test_case, batch, channel, height, width, placement, sbp):
    weight = random_tensor(1, channel).oneflow
    bias = random_tensor(1, channel).oneflow
    x = random_tensor(4, batch, channel, height, width).oneflow
    addend = random_tensor(4, batch, channel, height, width).oneflow

    params_sbp = [flow.sbp.broadcast for _ in range(len(sbp))]
    fused_bn = flow.nn.FusedBatchNorm2d(channel).to_global(
        placement=placement, sbp=params_sbp
    )
    fused_bn.eval()
    fused_bn.weight = flow.nn.Parameter(
        weight.to_global(placement=placement, sbp=params_sbp)
    )
    fused_bn.bias = flow.nn.Parameter(
        bias.to_global(placement=placement, sbp=params_sbp)
    )

    fused_x = x.to_global(placement=placement, sbp=sbp)
    fused_addend = addend.to_global(placement=placement, sbp=sbp)
    fused_out = fused_bn(fused_x, fused_addend)

    device = placement.type
    origin_bn = flow.nn.BatchNorm2d(channel).to(device)
    origin_bn.eval()
    origin_bn.weight = flow.nn.Parameter(weight.to_local().to(device))
    origin_bn.bias = flow.nn.Parameter(bias.to_local().to(device))

    origin_x = x.to_local().to(device)
    origin_addend = addend.to_local().to(device)
    origin_out = flow.nn.functional.relu(origin_bn(origin_x) + origin_addend)

    # test output.
    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )


def _test_bn_relu_eval(test_case, batch, channel, height, width, placement, sbp):
    weight = random_tensor(1, channel).oneflow
    bias = random_tensor(1, channel).oneflow
    x = random_tensor(4, batch, channel, height, width).oneflow

    params_sbp = [flow.sbp.broadcast for _ in range(len(sbp))]
    fused_bn = flow.nn.FusedBatchNorm2d(channel).to_global(
        placement=placement, sbp=params_sbp
    )
    fused_bn.eval()
    fused_bn.weight = flow.nn.Parameter(
        weight.to_global(placement=placement, sbp=params_sbp)
    )
    fused_bn.bias = flow.nn.Parameter(
        bias.to_global(placement=placement, sbp=params_sbp)
    )

    fused_x = x.to_global(placement=placement, sbp=sbp)
    fused_out = fused_bn(fused_x, None)

    device = placement.type
    origin_bn = flow.nn.BatchNorm2d(channel).to(device)
    origin_bn.eval()
    origin_bn.weight = flow.nn.Parameter(weight.to_local().to(device))
    origin_bn.bias = flow.nn.Parameter(bias.to_local().to(device))

    origin_x = x.to_local().to(device)
    origin_out = flow.nn.functional.relu(origin_bn(origin_x))

    # test output.
    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestBnAddRelu(flow.unittest.TestCase):
    @globaltest
    def test_bn_add_relu2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["batch"] = [8, 16, 64]
        arg_dict["channels"] = [8, 16]
        arg_dict["height"] = [8, 16]
        arg_dict["width"] = [8, 16]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                if placement.type != "cuda":
                    continue
                # mean and variance maybe inconsistent if input is split into each rank for training.
                # max_dim=0 will disable generating split sbp.
                for sbp in all_sbp(placement, max_dim=0):
                    _test_bn_add_relu(test_case, *arg, placement, sbp)
                    _test_bn_relu(test_case, *arg, placement, sbp)
                for sbp in all_sbp(placement, max_dim=1):
                    _test_bn_add_relu_eval(test_case, *arg, placement, sbp)
                    _test_bn_relu_eval(test_case, *arg, placement, sbp)


if __name__ == "__main__":
    unittest.main()
