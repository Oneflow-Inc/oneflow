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

import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *

import torch as pytorch_origin
import oneflow as oneflow_origin


def _test_convnd_grad_grad_impl(test_case, ndim, rtol=1e-4, atol=1e-5):
    minibatch = np.random.randint(1, 5)
    groups = np.random.randint(1, 5)
    in_channels = np.random.randint(1, 5) * groups
    out_channels = in_channels * np.random.randint(1, 5)
    padding = np.random.randint(1, 3)
    stride = np.random.randint(1, 3)
    dilation = np.random.randint(1, 3)

    x_shape = [minibatch, in_channels] + [np.random.randint(8, 12) for i in range(ndim)]
    w_shape = [out_channels, in_channels // groups] + [
        np.random.randint(2, 5) for i in range(ndim)
    ]

    x = random_tensor(len(x_shape), *x_shape)
    w = random_tensor(len(w_shape), *w_shape)
    init_grad_x = random_tensor(len(x_shape), *x_shape)
    init_grad_w = random_tensor(len(w_shape), *w_shape)

    y = eval(f"torch.nn.functional.conv{ndim}d")(
        x, w, stride=stride, padding=padding, groups=groups, dilation=dilation
    )
    init_grad_y = random_tensor(len(y.oneflow.shape), *y.oneflow.shape)

    dx = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=init_grad_y,
        create_graph=True,
        retain_graph=True,
    )[0]
    test_case.assertTrue(
        np.allclose(
            dx.pytorch.detach().cpu().numpy(),
            dx.oneflow.detach().numpy(),
            rtol=rtol,
            atol=atol,
        )
    )

    dw = torch.autograd.grad(
        outputs=y,
        inputs=w,
        grad_outputs=init_grad_y,
        create_graph=True,
        retain_graph=True,
    )[0]
    test_case.assertTrue(
        np.allclose(
            dw.pytorch.detach().cpu().numpy(),
            dw.oneflow.detach().numpy(),
            rtol=rtol,
            atol=atol,
        )
    )

    # torch.autograd.grad in autotest does not support inputs/outpus/grad_outputs as a list
    # so use the original pytorch/oneflow module
    ddx_pytorch, ddw_pytorch = pytorch_origin.autograd.grad(
        outputs=[dx.pytorch, dw.pytorch],
        inputs=[x.pytorch, w.pytorch],
        grad_outputs=[init_grad_x.pytorch, init_grad_w.pytorch],
        create_graph=True,
        retain_graph=True,
    )
    ddx_oneflow, ddw_oneflow = oneflow_origin.autograd.grad(
        outputs=[dx.oneflow, dw.oneflow],
        inputs=[x.oneflow, w.oneflow],
        grad_outputs=[init_grad_x.oneflow, init_grad_w.oneflow],
        create_graph=True,
        retain_graph=True,
    )

    test_case.assertTrue(
        np.allclose(
            ddw_pytorch.detach().cpu().numpy(),
            ddw_oneflow.detach().numpy(),
            rtol=rtol,
            atol=atol,
        )
    )
    test_case.assertTrue(
        np.allclose(
            ddx_pytorch.detach().cpu().numpy(),
            ddx_oneflow.detach().numpy(),
            rtol=rtol,
            atol=atol,
        )
    )

    dgrad_dx = torch.autograd.grad(
        outputs=dx,
        inputs=init_grad_y,
        grad_outputs=init_grad_x,
        create_graph=True,
        retain_graph=True,
    )[0]
    test_case.assertTrue(
        np.allclose(
            dgrad_dx.pytorch.detach().cpu().numpy(),
            dgrad_dx.oneflow.detach().numpy(),
            rtol=rtol,
            atol=atol,
        )
    )

    dgrad_dw = torch.autograd.grad(
        outputs=dw,
        inputs=init_grad_y,
        grad_outputs=init_grad_w,
        create_graph=True,
        retain_graph=True,
    )[0]
    test_case.assertTrue(
        np.allclose(
            dgrad_dw.pytorch.detach().cpu().numpy(),
            dgrad_dw.oneflow.detach().numpy(),
            rtol=rtol,
            atol=atol,
        )
    )


class TestConvHigherDerivative(flow.unittest.TestCase):
    def test_conv1d_grad_grad(test_case):
        _test_convnd_grad_grad_impl(test_case, 1)

    def test_conv2d_grad_grad(test_case):
        _test_convnd_grad_grad_impl(test_case, 2)

    def test_conv3d_grad_grad(test_case):
        _test_convnd_grad_grad_impl(test_case, 3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
