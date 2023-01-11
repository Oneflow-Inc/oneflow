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
import torch as pytorch
import torchvision
from oneflow.test_utils.automated_test_util import *


def _get_np_rois():
    random_img_idx = np.asarray(
        [random(0, 2).to(int).value() for _ in range(200)]
    ).reshape((200, 1))
    random_box_idx = np.asarray(
        [random(0, 64 * 64).to(float).value() for _ in range(400)]
    ).reshape((200, 2))

    def get_h_w(idx1, idx2):
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        h1 = idx1 // 64
        w1 = idx1 % 64
        h2 = idx2 // 64
        w2 = idx2 % 64
        return [x / 2 for x in [h1, w1, h2, w2]]

    zipped = zip(random_box_idx[:, 0], random_box_idx[:, 1])
    concated = [get_h_w(idx1, idx2) for (idx1, idx2) in zipped]
    concated = np.array(concated)
    rois = np.hstack((random_img_idx, concated)).astype(np.float32)
    return rois


def _test_roi_align(test_case, placement, rois_sbp):
    dims = [8, 8, 64, 64]
    x = random_tensor(4, *dims).to_global(
        placement=placement,
        sbp=[flow.sbp.broadcast for _ in range(len(placement.ranks.shape))],
    )
    x.oneflow = x.oneflow.detach().requires_grad_()
    x.pytorch = x.pytorch.detach().requires_grad_()

    def get_h_w(idx1, idx2):
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        h1 = idx1 // 64
        w1 = idx1 % 64
        h2 = idx2 // 64
        w2 = idx2 % 64
        return [x / 2 for x in [h1, w1, h2, w2]]

    np_rois = _get_np_rois()
    of_rois = (
        flow.tensor(np_rois, dtype=flow.float)
        .to_global(placement=flow.placement.all("cpu"), sbp=[flow.sbp.broadcast,])
        .to_global(placement, rois_sbp)
    )
    torch_rois = pytorch.tensor(np_rois)

    of_out = flow.roi_align(x.oneflow, of_rois, 2.0, 14, 14, 2, True)
    torch_out = torchvision.ops.roi_align(
        x.pytorch,
        torch_rois,
        spatial_scale=2.0,
        output_size=[14, 14],
        sampling_ratio=2,
        aligned=True,
    )

    # compare output
    of_local = of_out.to_global(
        placement=flow.placement.all("cpu"), sbp=[flow.sbp.broadcast,]
    ).to_local()
    test_case.assertTrue(
        np.allclose(
            of_local.numpy(), torch_out.detach().cpu().numpy(), rtol=1e-04, atol=1e-4
        )
    )

    # compare backward
    of_out.sum().backward()
    torch_out.sum().backward()
    of_input_grad = x.oneflow.grad.to_global(
        placement=flow.placement.all("cpu"), sbp=[flow.sbp.broadcast,]
    ).to_local()
    torch_input_grad = x.pytorch.grad.detach().cpu()
    test_case.assertTrue(
        np.allclose(
            of_input_grad.numpy(), torch_input_grad.numpy(), rtol=1e-04, atol=1e-4
        )
    )


def _test_roi_align_in_fixed_data_impl(test_case, placement, sbp):
    from test_roi_align import input_np, rois_np, input_grad_np

    input = (
        flow.tensor(input_np, dtype=flow.float32)
        .to_global(flow.placement.all("cpu"), [flow.sbp.broadcast,])
        .to_global(placement, sbp)
        .requires_grad_()
    )
    rois = (
        flow.tensor(rois_np, dtype=flow.float32)
        .to_global(flow.placement.all("cpu"), [flow.sbp.broadcast,])
        .to_global(
            placement, [flow.sbp.broadcast for _ in range(len(placement.ranks.shape))]
        )
    )
    of_out = flow.roi_align(input, rois, 2.0, 5, 5, 2, True)
    of_out.sum().backward()
    test_case.assertTrue(
        np.allclose(input.grad.numpy(), input_grad_np, rtol=1e-04, atol=1e-4)
    )


class TestGlobalRoiAlign(flow.unittest.TestCase):
    # TODO(wyg): It is a bug in pytorch-1.9.0, torchvision-0.10.0 and python3.7.10.
    #            Open this test after updating the versions of pytorch in CI.

    #  @globaltest
    #  def test_global_roi_align(test_case):
    #      for placement in all_placement():
    #          # TODO: roi_align only support gpu
    #          if placement.type == "cpu":
    #              continue
    #          for rois_sbp in all_sbp(placement, max_dim=0, except_partial_sum=True):
    #              _test_roi_align(test_case, placement, rois_sbp)

    def test_global_roi_align_in_fixed_data(test_case):
        for placement in all_placement():
            # TODO: roi_align only support gpu
            if placement.type == "cpu":
                continue
            for rois_sbp in all_sbp(placement, max_dim=0, except_partial_sum=True):
                _test_roi_align_in_fixed_data_impl(test_case, placement, rois_sbp)


if __name__ == "__main__":
    unittest.main()
