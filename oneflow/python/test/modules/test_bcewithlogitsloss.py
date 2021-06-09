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
import oneflow.experimental as flow
import unittest
import numpy as np


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestBCEWithLogitsLossModule(flow.unittest.TestCase):
    def np_bcewithlogitsloss(output, target, weight=None, pos_weight=None, reduction='none'):
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
            # Do no reduction
            return _weighted_loss
            #     return _weighted_loss

    def test_BCEWithLogitsLoss_none(test_case):
        x = np.array(
            [[1.2, 0.2, -0.3],
             [0.7, 0.6, -2]]
        ).astype(np.float32)
        y = np.array([[0, 1, 0],
                              [1, 0, 1]]).astype(np.int)
        input = flow.Tensor(x, dtype=flow.float32)

        target = flow.Tensor(y, dtype=flow.int64)
        bcewithlogits_loss = flow.nn.BCEWithLogitsLoss(reduction="none")
        of_out = bcewithlogits_loss(input, target)
        # np_out = nll_loss_1d(input.numpy(), target.numpy(), reduction="mean")
        import torch
        criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        torch_out = criterion(input.numpy(),target.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), torch_out))

if __name__ == "__main__":
    unittest.main()
