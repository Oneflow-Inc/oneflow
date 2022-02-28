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


def _test_bn_add_relu(test_case, device, batch, channel, height, width):
    weight_numpy = np.random.randn(channel)
    bias_numpy = np.random.randn(channel)

    fused_x = np.random.randn(batch, channel, height, width)
    fused_x_tensor = flow.Tensor(fused_x).to(device)
    fused_x_tensor.requires_grad = True

    fused_addend = np.random.randn(batch, channel, height, width)
    fused_addend_tensor = flow.Tensor(fused_addend).to(device)
    fused_addend_tensor.requires_grad = True

    fused_weight_tensor = flow.nn.Parameter(flow.Tensor(weight_numpy).to(device))
    fused_bias_tensor = flow.nn.Parameter(flow.Tensor(bias_numpy).to(device))

    fused_bn = flow.nn.FusedBatchNorm2d(channel).to(device)
    fused_bn.weight = fused_weight_tensor
    fused_bn.bias = fused_bias_tensor
    fused_out = fused_bn(fused_x_tensor, fused_addend_tensor)

    origin_x_tensor = flow.Tensor(fused_x).to(device)
    origin_x_tensor.requires_grad = True

    origin_addend_tensor = flow.Tensor(fused_addend).to(device)
    origin_addend_tensor.requires_grad = True

    origin_weight_tensor = flow.nn.Parameter(flow.Tensor(weight_numpy).to(device))
    origin_bias_tensor = flow.nn.Parameter(flow.Tensor(bias_numpy).to(device))

    origin_batch_norm = flow.nn.BatchNorm2d(channel).to(device)
    origin_batch_norm.weight = origin_weight_tensor
    origin_batch_norm.bias = origin_bias_tensor

    origin_out = origin_batch_norm(origin_x_tensor) + origin_addend_tensor
    origin_out = flow.nn.functional.relu(origin_out)

    total_out = fused_out + origin_out
    total_out_sum = total_out.sum()

    total_out_sum.backward()

    # test output.
    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )
    # test input grad.
    test_case.assertTrue(
        np.allclose(
            fused_x_tensor.grad.numpy(),
            origin_x_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_addend_tensor.grad.numpy(),
            origin_addend_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    # test weight and bias grad.
    test_case.assertTrue(
        np.allclose(
            fused_weight_tensor.grad.numpy(),
            origin_weight_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_bias_tensor.grad.numpy(),
            origin_bias_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    # test running mean and running variance.
    test_case.assertTrue(
        np.allclose(
            fused_bn.running_mean.numpy(),
            origin_batch_norm.running_mean.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_bn.running_var.numpy(),
            origin_batch_norm.running_var.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )


def _test_bn_relu(test_case, device, batch, channel, height, width):
    weight_numpy = np.random.randn(channel)
    bias_numpy = np.random.randn(channel)

    fused_x = np.random.randn(batch, channel, height, width)
    fused_x_tensor = flow.Tensor(fused_x).to(device)
    fused_x_tensor.requires_grad = True

    fused_weight_tensor = flow.nn.Parameter(flow.Tensor(weight_numpy).to(device))
    fused_bias_tensor = flow.nn.Parameter(flow.Tensor(bias_numpy).to(device))

    fused_bn = flow.nn.FusedBatchNorm2d(channel).to(device)
    fused_bn.weight = fused_weight_tensor
    fused_bn.bias = fused_bias_tensor
    fused_out = fused_bn(fused_x_tensor, None)

    origin_x_tensor = flow.Tensor(fused_x).to(device)
    origin_x_tensor.requires_grad = True

    origin_weight_tensor = flow.nn.Parameter(flow.Tensor(weight_numpy).to(device))
    origin_bias_tensor = flow.nn.Parameter(flow.Tensor(bias_numpy).to(device))

    origin_batch_norm = flow.nn.BatchNorm2d(channel).to(device)
    origin_batch_norm.weight = origin_weight_tensor
    origin_batch_norm.bias = origin_bias_tensor

    origin_out = origin_batch_norm(origin_x_tensor)
    origin_out = flow.nn.functional.relu(origin_out)

    total_out = fused_out + origin_out
    total_out_sum = total_out.sum()

    total_out_sum.backward()

    # test output.
    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )
    # test input grad.
    test_case.assertTrue(
        np.allclose(
            fused_x_tensor.grad.numpy(),
            origin_x_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )

    # test weight and bias grad.
    test_case.assertTrue(
        np.allclose(
            fused_weight_tensor.grad.numpy(),
            origin_weight_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_bias_tensor.grad.numpy(),
            origin_bias_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    # test running mean and running variance.
    test_case.assertTrue(
        np.allclose(
            fused_bn.running_mean.numpy(),
            origin_batch_norm.running_mean.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_bn.running_var.numpy(),
            origin_batch_norm.running_var.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )


def _test_bn_relu_track_running_states_false(
    test_case, device, batch, channel, height, width
):
    weight_numpy = np.random.randn(channel)
    bias_numpy = np.random.randn(channel)

    fused_x = np.random.randn(batch, channel, height, width)
    fused_x_tensor = flow.Tensor(fused_x).to(device)
    fused_x_tensor.requires_grad = True

    fused_weight_tensor = flow.nn.Parameter(flow.Tensor(weight_numpy).to(device))
    fused_bias_tensor = flow.nn.Parameter(flow.Tensor(bias_numpy).to(device))

    fused_bn = flow.nn.FusedBatchNorm2d(channel, track_running_stats=False).to(device)
    fused_bn.weight = fused_weight_tensor
    fused_bn.bias = fused_bias_tensor
    fused_out = fused_bn(fused_x_tensor, None)

    origin_x_tensor = flow.Tensor(fused_x).to(device)
    origin_x_tensor.requires_grad = True

    origin_weight_tensor = flow.nn.Parameter(flow.Tensor(weight_numpy).to(device))
    origin_bias_tensor = flow.nn.Parameter(flow.Tensor(bias_numpy).to(device))

    origin_batch_norm = flow.nn.BatchNorm2d(channel, track_running_stats=False).to(
        device
    )
    origin_batch_norm.weight = origin_weight_tensor
    origin_batch_norm.bias = origin_bias_tensor

    origin_out = origin_batch_norm(origin_x_tensor)
    origin_out = flow.nn.functional.relu(origin_out)

    total_out = fused_out + origin_out
    total_out_sum = total_out.sum()

    total_out_sum.backward()

    # test output.
    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )
    # test input grad.
    test_case.assertTrue(
        np.allclose(
            fused_x_tensor.grad.numpy(),
            origin_x_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    # test weight and bias grad.
    test_case.assertTrue(
        np.allclose(
            fused_weight_tensor.grad.numpy(),
            origin_weight_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_bias_tensor.grad.numpy(),
            origin_bias_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    # When track running states is False, the running mean and running variance will be set as None.
    test_case.assertIsNone(fused_bn.running_mean)
    test_case.assertIsNone(origin_batch_norm.running_mean)
    test_case.assertIsNone(fused_bn.running_var)
    test_case.assertIsNone(origin_batch_norm.running_var)


def _test_bn_add_relu_track_running_states_false(
    test_case, device, batch, channel, height, width
):
    weight_numpy = np.random.randn(channel)
    bias_numpy = np.random.randn(channel)

    fused_x = np.random.randn(batch, channel, height, width)
    fused_x_tensor = flow.Tensor(fused_x).to(device)
    fused_x_tensor.requires_grad = True

    fused_addend = np.random.randn(batch, channel, height, width)
    fused_addend_tensor = flow.Tensor(fused_addend).to(device)
    fused_addend_tensor.requires_grad = True

    fused_weight_tensor = flow.nn.Parameter(flow.Tensor(weight_numpy).to(device))
    fused_bias_tensor = flow.nn.Parameter(flow.Tensor(bias_numpy).to(device))

    fused_bn = flow.nn.FusedBatchNorm2d(channel, track_running_stats=False).to(device)
    fused_bn.weight = fused_weight_tensor
    fused_bn.bias = fused_bias_tensor
    fused_out = fused_bn(fused_x_tensor, fused_addend_tensor)

    origin_x_tensor = flow.Tensor(fused_x).to(device)
    origin_x_tensor.requires_grad = True

    origin_addend_tensor = flow.Tensor(fused_addend).to(device)
    origin_addend_tensor.requires_grad = True

    origin_weight_tensor = flow.nn.Parameter(flow.Tensor(weight_numpy).to(device))
    origin_bias_tensor = flow.nn.Parameter(flow.Tensor(bias_numpy).to(device))

    origin_batch_norm = flow.nn.BatchNorm2d(channel, track_running_stats=False).to(
        device
    )
    origin_batch_norm.weight = origin_weight_tensor
    origin_batch_norm.bias = origin_bias_tensor

    origin_out = origin_batch_norm(origin_x_tensor) + origin_addend_tensor
    origin_out = flow.nn.functional.relu(origin_out)

    total_out = fused_out + origin_out
    total_out_sum = total_out.sum()

    total_out_sum.backward()

    # test output.
    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )
    # test input grad.
    test_case.assertTrue(
        np.allclose(
            fused_x_tensor.grad.numpy(),
            origin_x_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_addend_tensor.grad.numpy(),
            origin_addend_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    # test weight and bias grad.
    test_case.assertTrue(
        np.allclose(
            fused_weight_tensor.grad.numpy(),
            origin_weight_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            fused_bias_tensor.grad.numpy(),
            origin_bias_tensor.grad.numpy(),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    # When track running states is False, the running mean and running variance will be set as None.
    test_case.assertIsNone(fused_bn.running_mean)
    test_case.assertIsNone(origin_batch_norm.running_mean)
    test_case.assertIsNone(fused_bn.running_var)
    test_case.assertIsNone(origin_batch_norm.running_var)


def _test_bn_add_relu_eval(test_case, device, batch, channel, height, width):
    weight_numpy = np.random.randn(channel)
    bias_numpy = np.random.randn(channel)

    fused_x = np.random.randn(batch, channel, height, width)
    fused_x_tensor = flow.Tensor(fused_x).to(device)

    fused_addend = np.random.randn(batch, channel, height, width)
    fused_addend_tensor = flow.Tensor(fused_addend).to(device)

    fused_weight_tensor = flow.nn.Parameter(flow.Tensor(weight_numpy).to(device))
    fused_bias_tensor = flow.nn.Parameter(flow.Tensor(bias_numpy).to(device))

    fused_bn = flow.nn.FusedBatchNorm2d(channel).to(device)
    fused_bn.eval()
    fused_bn.weight = fused_weight_tensor
    fused_bn.bias = fused_bias_tensor
    fused_out = fused_bn(fused_x_tensor, fused_addend_tensor)

    origin_x_tensor = flow.Tensor(fused_x).to(device)

    origin_addend_tensor = flow.Tensor(fused_addend).to(device)

    origin_weight_tensor = flow.nn.Parameter(flow.Tensor(weight_numpy).to(device))
    origin_bias_tensor = flow.nn.Parameter(flow.Tensor(bias_numpy).to(device))

    origin_batch_norm = flow.nn.BatchNorm2d(channel).to(device)
    origin_batch_norm.eval()
    origin_batch_norm.weight = origin_weight_tensor
    origin_batch_norm.bias = origin_bias_tensor

    origin_out = origin_batch_norm(origin_x_tensor) + origin_addend_tensor
    origin_out = flow.nn.functional.relu(origin_out)

    # test output.
    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )


def _test_bn_relu_eval(test_case, device, batch, channel, height, width):
    weight_numpy = np.random.randn(channel)
    bias_numpy = np.random.randn(channel)

    fused_x = np.random.randn(batch, channel, height, width)
    fused_x_tensor = flow.Tensor(fused_x).to(device)

    fused_weight_tensor = flow.nn.Parameter(flow.Tensor(weight_numpy).to(device))
    fused_bias_tensor = flow.nn.Parameter(flow.Tensor(bias_numpy).to(device))

    fused_bn = flow.nn.FusedBatchNorm2d(channel).to(device)
    fused_bn.eval()
    fused_bn.weight = fused_weight_tensor
    fused_bn.bias = fused_bias_tensor
    fused_out = fused_bn(fused_x_tensor)

    origin_x_tensor = flow.Tensor(fused_x).to(device)

    origin_weight_tensor = flow.nn.Parameter(flow.Tensor(weight_numpy).to(device))
    origin_bias_tensor = flow.nn.Parameter(flow.Tensor(bias_numpy).to(device))

    origin_batch_norm = flow.nn.BatchNorm2d(channel).to(device)
    origin_batch_norm.eval()
    origin_batch_norm.weight = origin_weight_tensor
    origin_batch_norm.bias = origin_bias_tensor

    origin_out = origin_batch_norm(origin_x_tensor)
    origin_out = flow.nn.functional.relu(origin_out)

    # test output.
    test_case.assertTrue(
        np.allclose(fused_out.numpy(), origin_out.numpy(), atol=1e-4, rtol=1e-4)
    )


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test gpu cases")
class TestBnAddRelu(flow.unittest.TestCase):
    def test_bn_add_relu2d(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_bn_add_relu,
            _test_bn_relu,
            _test_bn_relu_track_running_states_false,
            _test_bn_add_relu_track_running_states_false,
            _test_bn_add_relu_eval,
            _test_bn_relu_eval,
        ]
        arg_dict["device"] = ["cuda"]
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            arg_dict["device"] = ["cpu"]
        arg_dict["batch"] = [1, 2, 8]
        arg_dict["channels"] = [4, 6]
        arg_dict["height"] = [6, 8]
        arg_dict["width"] = [12, 8]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
