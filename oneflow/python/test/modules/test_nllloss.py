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


def np_nll_loss(input, target, mode=None):
    n = input.shape[0]
    c = input.shape[1]
    input = -input
    mask = target[0:n]

    if len(mask.shape) > 1:
            input = [input[i, int(mask[i][0]),] for i in range(n)]
        else:
            input = [input[i, int(mask[i]),] for i in range(n)]
    
    if mode == "sum":
        loss = 0
        for x in input:
            loss += x
        return loss
    elif mode == "mean":
        loss = 0
        for x in input:
            loss += x
        return loss / n
    else:
        new_shape = tuple()
        if len(mask.shape) == 1:
            input = np.reshape(input, (target.shape[0], 1,))
        else:
            new_shape.append(target.shape[0])
            new_shape.append(1)
            for i in range(1, len(target.shape)):
                new_shape.append(target.shape[i])
            input = np.reshape(input, new_shape)
        return input


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestNLLLossModule(flow.unittest.TestCase):
    def test_nllloss_none(test_case):
        x = np.array(
            [
                [0.88103855, 0.9908683, 0.6226845],
                [0.53331435, 0.07999352, 0.8549948],
                [0.25879037, 0.39530203, 0.698465],
                [0.73427284, 0.63575995, 0.18827209],
                [0.05689114, 0.0862954, 0.6325046],
            ]
        ).astype(np.float32)
        y = np.array([0, 2, 1, 1, 0]).astype(np.int)
        input = flow.Tensor(x, dtype=flow.float32)

        target = flow.Tensor(y, dtype=flow.int)
        nll_loss = flow.nn.NLLLoss()
        of_out = nll_loss(input, target)
        np_out = np_nll_loss(input.numpy(), target.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_nllloss_mean(test_case):
        x = np.array(
            [
                [0.88103855, 0.9908683, 0.6226845],
                [0.53331435, 0.07999352, 0.8549948],
                [0.25879037, 0.39530203, 0.698465],
                [0.73427284, 0.63575995, 0.18827209],
                [0.05689114, 0.0862954, 0.6325046],
            ]
        ).astype(np.float32)
        y = np.array([0, 2, 1, 1, 0]).astype(np.int)
        input = flow.Tensor(x, dtype=flow.float32)

        target = flow.Tensor(y, dtype=flow.int)
        nll_loss = flow.nn.NLLLoss(reduction="mean")
        of_out = nll_loss(input, target)
        np_out = np_nll_loss(input.numpy(), target.numpy(), mode="mean")
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_nllloss_sum(test_case):
        x = np.array(
            [
                [0.88103855, 0.9908683, 0.6226845],
                [0.53331435, 0.07999352, 0.8549948],
                [0.25879037, 0.39530203, 0.698465],
                [0.73427284, 0.63575995, 0.18827209],
                [0.05689114, 0.0862954, 0.6325046],
            ]
        ).astype(np.float32)
        y = np.array([0, 2, 1, 1, 0]).astype(np.int)
        input = flow.Tensor(x, dtype=flow.float32)

        target = flow.Tensor(y, dtype=flow.int)
        nll_loss = flow.nn.NLLLoss(reduction="sum")
        of_out = nll_loss(input, target)
        np_out = np_nll_loss(input.numpy(), target.numpy(), mode="sum")
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_nllloss_segmentation_none(test_case):
        x = np.array(
            [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
        ).astype(np.float32)
        input = flow.Tensor(x, dtype=flow.float32)
        y = np.array([[[1, 0], [0, 1]]]).astype(np.int)
        target = flow.Tensor(y, dtype=flow.int)
        nll_loss = flow.nn.NLLLoss()
        of_out = nll_loss(input, target)
        np_out = np_nll_loss(input.numpy(), target.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


if __name__ == "__main__":
    unittest.main()
