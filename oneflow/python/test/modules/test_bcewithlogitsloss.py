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

import oneflow.experimental as flow


def np_bcewithlogitsloss(output, target, weight=None, pos_weight=None, reduction='none'):
    if input_shape_len >= 5:
        raise NotImplemented

    _neg_input = np.negative(output)
    _max_val = np.clip(_neg_input, 0, None)
    _neg_max_val = np.negative(_max_val)

    if pos_weight is not None:
        assert pos_weight.shape[0] == output.shape[-1], (
            "The length of `pos_weight` must be equal to the number of classes. "
            "Found the length of pos_weight {} vs classes {}".format(
                pos_weight.shape[0], output.shape[-1]
            )
        )
        _log_weight = ((pos_weight - 1) * target) + 1
        _loss = (1 - target) * output + _log_weight * (
                np.log(
                    np.exp(_neg_max_val) + np.exp(_neg_input - _max_val)
                )
                + _max_val
        )
    else:
        _loss = (1 - target) * output + _max_val
        _loss += np.log(
            np.exp(_neg_max_val) + np.exp(_neg_input - _max_val)
        )

    if weight is not None:
        assert (
                weight.shape == output.shape
        ), "The weight shape must be the same as Input shape"
        _weighted_loss = weight * _loss
    else:
        _weighted_loss = _loss

    if reduction == "mean":
        return _weighted_loss.mean()
    elif reduction == "sum":
        return _weighted_loss.sum()
    else:
        return _weighted_loss


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestBCEWithLogitsLossModule(flow.unittest.TestCase):
    def test_bcewithlogitsloss_none(test_case):
        x = np.random.randn(2,3).astype(np.float32)
        y = np.random.randint(0,2,[2,3]).astype(np.float32)
        w = np.random.randn(2,3).astype(np.float32)
        pw = np.random.randn(3).astype(np.float32)

        input = flow.Tensor(x, dtype=flow.float32)
        target = flow.Tensor(y, dtype=flow.float32)
        weight = flow.Tensor(w, dtype=flow.float32)
        pos_weight = flow.Tensor(pw, dtype=flow.float32)

        bcewithlogits_loss = flow.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction="none")
        of_out = bcewithlogits_loss(input, target)

        np_out = np_bcewithlogitsloss(x, y, weight=w, pos_weight=pw, reduction='none')
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out,  1e-4, 1e-4))

    def test_bcewithlogitsloss_mean(test_case):
        x = np.random.randn(2,3,3).astype(np.float32)
        y = np.random.randint(0,2,[2,3,3]).astype(np.float32)
        w = np.random.randn(2,3,3).astype(np.float32)
        pw = np.random.randn(3).astype(np.float32)

        input = flow.Tensor(x, dtype=flow.float32)
        target = flow.Tensor(y, dtype=flow.float32)
        weight = flow.Tensor(w, dtype=flow.float32)
        pos_weight = flow.Tensor(pw, dtype=flow.float32)

        bcewithlogits_loss = flow.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction="mean")
        of_out = bcewithlogits_loss(input, target)

        np_out = np_bcewithlogitsloss(x, y, weight=w, pos_weight=pw, reduction='mean')
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out,  1e-4, 1e-4))

    def test_bcewithlogitsloss_sum(test_case):
        x = np.random.randn(2,3,4,5).astype(np.float32)
        y = np.random.randint(0,2,[2,3,4,5]).astype(np.float32)
        w = np.random.randn(2,3,4,5).astype(np.float32)
        pw = np.random.randn(5).astype(np.float32)

        input = flow.Tensor(x, dtype=flow.float32)
        target = flow.Tensor(y, dtype=flow.float32)
        weight = flow.Tensor(w, dtype=flow.float32)
        pos_weight = flow.Tensor(pw, dtype=flow.float32)

        bcewithlogits_loss = flow.nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight, reduction="sum")
        of_out = bcewithlogits_loss(input, target)

        np_out = np_bcewithlogitsloss(x, y, weight=w, pos_weight=pw, reduction='sum')
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out,  1e-4, 1e-4))


if __name__ == "__main__":
    unittest.main()
