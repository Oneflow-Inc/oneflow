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
    mask = np.array(target[0:n])

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
        return np.concatenate(input,)


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_nllloss(test_case):
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
        print(of_out)


if __name__ == "__main__":
    unittest.main()
