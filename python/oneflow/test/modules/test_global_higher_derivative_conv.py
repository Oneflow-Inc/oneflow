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


def _test_convnd_grad_grad_impl(test_case, ndim, placement, x_sbp, w_sbp):
    # print(placement, x_sbp, w_sbp)
    x_data_shape = [16 for i in range(ndim)]
    w_data_shape = [8 for i in range(ndim)]
    x_shape = [16, 8, *x_data_shape]
    w_shape = [8, 8, *w_data_shape]
    grad_shape = [16, 8, 9, 9]

    x = (
        torch.tensor(np.random.rand(*x_shape))
        .requires_grad_(True)
        .to_global(placement=placement, sbp=x_sbp)
    )
    w = (
        torch.tensor(np.random.rand(*w_shape))
        .requires_grad_(True)
        .to_global(placement=placement, sbp=flow.sbp.broadcast)
    )
    init_grad_y = (
        torch.tensor(np.random.rand(*grad_shape))
        .requires_grad_(True)
        .to_global(placement=placement, sbp=x_sbp)
    )
    init_grad_x = torch.tensor(np.random.rand(*x.oneflow.shape)).to_global(
        placement=placement, sbp=x_sbp
    )
    init_grad_w = torch.tensor(np.random.rand(*w.oneflow.shape)).to_global(
        placement=placement, sbp=flow.sbp.broadcast
    )

    y = eval(f"torch.nn.functional.conv{ndim}d")(
        x, w, stride=1, padding=0, groups=1, dilation=1
    )
    # init_grad_y.to_global(placement=placement, sbp=y.sbp)
    # print(init_grad_y.pytorch.shape, y.pytorch.shape, x.pytorch.shape)
    assert init_grad_y.pytorch.shape == y.pytorch.shape
    dx = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=init_grad_y,
        create_graph=True,
        retain_graph=True,
    )[0]

    test_case.assertTrue(
        np.allclose(
            dx.pytorch.detach().cpu().numpy(), dx.oneflow.to_local().detach().numpy()
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
        np.allclose(dw.pytorch.detach().cpu().numpy(), dw.oneflow.detach().numpy())
    )

    # autotest torch.autograd.grad 不支持 inputs/outpus/grad_outputs 为 list
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
        np.allclose(ddw_pytorch.detach().cpu().numpy(), ddw_oneflow.detach().numpy())
    )
    test_case.assertTrue(
        np.allclose(ddx_pytorch.detach().cpu().numpy(), ddx_oneflow.detach().numpy())
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
            dgrad_dx.pytorch.detach().cpu().numpy(), dgrad_dx.oneflow.detach().numpy()
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
            dgrad_dw.pytorch.detach().cpu().numpy(), dgrad_dw.oneflow.detach().numpy()
        )
    )


class TestGlobalConvHigherDerivative(flow.unittest.TestCase):
    @globaltest
    def test_conv1d_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_convnd_grad_grad_impl(test_case, 1, placement, sbp)

    @globaltest
    def test_conv2d_grad_grad(test_case):
        for placement in all_placement():
            for x_sbp in all_sbp(placement, max_dim=1):
                # for w_sbp in all_sbp(placement, max_dim=2):
                _test_convnd_grad_grad_impl(
                    test_case, 2, placement, x_sbp, flow.sbp.broadcast
                )

    @globaltest
    def test_conv3d_grad_grad(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                _test_convnd_grad_grad_impl(test_case, 3, placement, sbp)


if __name__ == "__main__":
    unittest.main()
