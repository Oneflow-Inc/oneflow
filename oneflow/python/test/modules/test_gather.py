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


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestGather(flow.unittest.TestCase):
    def test_gather(test_case):
        input = np.array([[1, 2], [3, 4]])
        index = np.array([[0, 0], [1, 0]])
        np_out = np.take_along_axis(input, index, 0)
        output = flow.gather(
            flow.Tensor(input), flow.Tensor(index, dtype=flow.int), dim=0
        )
        test_case.assertTrue(np.array_equal(output.numpy(), np_out))

    def test_gather_tensor_function(test_case):
        input = np.array([[1, 2], [3, 4]])
        index = np.array([[0, 0], [1, 0]])
        np_out = np.take_along_axis(input, index, 1)
        input = flow.Tensor(input)
        index = flow.Tensor(index, dtype=flow.int)
        output = input.gather(index, dim=1)

        test_case.assertTrue(np.array_equal(output.numpy(), np_out))

    def test_gather_random_array(test_case):
        input = np.random.randn(3, 4, 3, 5)
        index = np.random.choice(np.arange(3), size=180, replace=True).reshape(
            (3, 4, 3, 5)
        )
        np_out = np.take_along_axis(input, index, 1)
        output = flow.gather(
            flow.Tensor(input), flow.Tensor(index, dtype=flow.int), dim=1
        )
        test_case.assertTrue(np.allclose(output.numpy(), np_out))

        np_out2 = np.take_along_axis(input, index, 2)
        output2 = flow.gather(
            flow.Tensor(input), flow.Tensor(index, dtype=flow.int), dim=2
        )
        test_case.assertTrue(np.allclose(output2.numpy(), np_out2))

        np_out3 = np.take_along_axis(input, index, 3)
        output3 = flow.gather(
            flow.Tensor(input), flow.Tensor(index, dtype=flow.int), dim=3
        )
        test_case.assertTrue(np.allclose(output3.numpy(), np_out3))


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestGatherBackward(flow.unittest.TestCase):
    def test_gather_backward(test_case):
        input = np.array([[1, 2], [3, 4]])
        index = np.array([[0, 0], [1, 0]])
        np_out = np.take_along_axis(input, index, 0)
        np_grad = _scatter_add_numpy(np.ones_like(np_out), 0, index, input.shape)

        of_input = flow.Tensor(input, requires_grad=True)
        output = flow.gather(of_input, flow.Tensor(index, dtype=flow.int), dim=0)
        out_sum = output.sum()
        out_sum.backward()

        test_case.assertTrue(np.array_equal(output.numpy(), np_out))
        test_case.assertTrue(np.array_equal(of_input.grad.numpy(), np_grad))


if __name__ == "__main__":
    unittest.main()
