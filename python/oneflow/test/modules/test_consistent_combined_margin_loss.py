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


def _scatter_add_numpy(src, dim, index, outshape):
    output = np.zeros(outshape)
    for srcidx in range(0, src.size):
        outcoord = np.unravel_index(srcidx, src.shape)
        outcoord = [*outcoord]
        outcoord[dim] = index[np.unravel_index(srcidx, index.shape)]
        output_offset = np.ravel_multi_index(outcoord, outshape)
        output[np.unravel_index(output_offset, outshape)] += src[
            np.unravel_index(srcidx, src.shape)
        ]
    return output


def _np_one_hot(indices, depth):
    return np.eye(depth)[indices.reshape(-1)]


def _np_gather_with_batch_dims(params, indices, axis):
    batch_dims = 1
    result = []
    for p, i in zip(params, indices):
        r = np.take_along_axis(p, i, axis - batch_dims)
        result.append(r)
    return np.stack(result)


def _np_gather_with_batch_dims_grad(params, indices, axis, output):
    batch_dims = 1
    result = []
    for p, i, o in zip(params, indices, output):
        r = _scatter_add_numpy(np.ones_like(o), axis - batch_dims, i, p.shape)
        result.append(r)
    return np.stack(result)


def _np_combined_margin_loss(np_input, np_label, m1, m2, m3):
    class_num = np_input.shape[1]
    if m1 != 1.0 or m2 != 0.0 or m3 != 0.0:
        if m1 == 1.0 and m2 == 0.0:
            gt_one_hot = _np_one_hot(np_label, class_num) * m3
            np_input = np_input - gt_one_hot
        else:
            np_label_expand = np.reshape(np_label, (np_label.shape[0], 1))
            zy = _np_gather_with_batch_dims(np_input, np_label_expand, 0)
            cos_t = zy * 1
            t = np.arccos(cos_t)
            if m1 != 1.0:
                t = t * m1
            if m2 > 0.0:
                t = t + m2
            body = np.cos(t)
            if m3 > 0.0:
                body = body - m3
            new_zy = body
            diff = new_zy - zy
            gt_one_hot = _np_one_hot(np_label, class_num)
            body = gt_one_hot * diff
            np_input = np_input + body
    return np_input


def _np_combined_margin_loss_grad(np_input, np_label, m1, m2, m3):
    class_num = np_input.shape[1]
    if m1 != 1.0 or m2 != 0.0 or m3 != 0.0:
        if m1 == 1.0 and m2 == 0.0:
            result = np.ones(np_input.shape)
        else:
            np_label_expand = np.reshape(np_label, (np_label.shape[0], 1))
            zy = _np_gather_with_batch_dims(np_input, np_label_expand, 0)
            dzy = _np_gather_with_batch_dims_grad(np_input, np_label_expand, 0, zy)
            cos_t = zy * 1
            t = np.arccos(cos_t)
            dt = -1 / np.sqrt((1 - cos_t * cos_t)) * dzy
            if m1 != 1.0:
                t = t * m1
                dt = dt * m1
            if m2 > 0.0:
                t = t + m2
            body = np.cos(t)
            dbody = -np.sin(t) * dt
            if m3 > 0.0:
                body = body - m3
            new_zy = body
            diff = new_zy - zy
            ddiff = dbody - dzy
            gt_one_hot = _np_one_hot(np_label, class_num)
            body = gt_one_hot * diff
            dbody = gt_one_hot * ddiff
            np_input = np_input + body
            result = np.ones(np_input.shape) + dbody
    else:
        result = np.ones(np_input.shape)
    return result


def _test_combined_margin_loss(test_case, placement, sbp):
    x = random_tensor(ndim=2, dim0=64, dim1=1000, low=-1, high=1).oneflow
    x = x.to_global(placement, sbp)
    x.retain_grad()
    labels = random_tensor(ndim=1, dim0=64, low=0, high=1000, dtype=int).oneflow
    labels = labels.to(flow.int).to_global(
        placement=placement, sbp=random_sbp(placement, max_dim=1).value()
    )

    np_x = x.numpy()
    np_labels = labels.numpy()

    m1 = oneof(0.3, 1.0).value()
    m2 = oneof(0.5, 0.0).value()
    m3 = oneof(0.4, 0.0).value()
    loss_func = flow.nn.CombinedMarginLoss(m1, m2, m3)
    output = loss_func(x, labels)
    output.sum().backward()

    output_ref = _np_combined_margin_loss(np_x, np_labels, m1, m2, m3)
    input_grad_ref = _np_combined_margin_loss_grad(np_x, np_labels, m1, m2, m3)

    test_case.assertTrue(np.allclose(output.numpy(), output_ref, rtol=1e-5, atol=1e-5))
    test_case.assertTrue(
        np.allclose(x.grad.numpy(), input_grad_ref, rtol=1e-4, atol=1e-4)
    )


class TestConsistentCombinedMarginLoss(flow.unittest.TestCase):
    @globaltest
    def test_combined_margin_loss(test_case):
        for placement in all_placement():
            for sbp in all_sbp(placement, max_dim=2):
                # combined margin loss op only support 1d sbp
                if len(sbp) != 1:
                    continue
                _test_combined_margin_loss(test_case, placement, sbp)


if __name__ == "__main__":
    unittest.main()
