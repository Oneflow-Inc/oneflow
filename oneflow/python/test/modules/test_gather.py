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


def gather_numpy(input, index, dim):
    """ 
    Gathers values along an axis specified by dim. 
    For a 3-D tensor the output is specified by: 
     out[i][j][k] = input[index[i][j][k]][j][k] # if dim == 0 
     out[i][j][k] = input[i][index[i][j][k]][k] # if dim == 1 
     out[i][j][k] = input[i][j][index[i][j][k]] # if dim == 2 

    :param dim: The axis along which to index 
    :param index: A tensor of indices of elements to gather 
    :return: tensor of gathered values 
    """
    idx_xsection_shape = index.shape[:dim] + index.shape[dim + 1 :]
    self_xsection_shape = input.shape[:dim] + input.shape[dim + 1 :]
    if idx_xsection_shape != self_xsection_shape:
        raise ValueError(
            "Except for dimension "
            + str(dim)
            + ", all dimensions of index and self should be the same size"
        )
    if index.dtype != np.dtype("int_"):
        raise TypeError("The values of index must be integers")
    data_swaped = np.swapaxes(input, 0, dim)
    index_swaped = np.swapaxes(index, 0, dim)
    gathered = np.choose(index_swaped, data_swaped)
    return np.swapaxes(gathered, 0, dim)


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestGather(flow.unittest.TestCase):
    def test_gather(test_case):
        input = np.array([[1, 2], [3, 4]])
        index = np.array([[0, 0], [1, 0]])
        np_out = gather_numpy(input, index, dim=0)
        output = flow.tmp.gather(
            flow.Tensor(input), flow.Tensor(index, dtype=flow.int), dim=0
        )
        test_case.assertTrue(np.array_equal(output.numpy(), np_out))

    def test_gather_random_array(test_case):
        input = np.random.randn(3, 4, 3, 5)
        index = np.random.choice(np.arange(3), size=180, replace=True).reshape(
            (3, 4, 3, 5)
        )
        np_out = gather_numpy(input, index, dim=1)
        output = flow.tmp.gather(
            flow.Tensor(input), flow.Tensor(index, dtype=flow.int), dim=1
        )
        test_case.assertTrue(np.allclose(output.numpy(), np_out))

        np_out2 = gather_numpy(input, index, dim=2)
        output2 = flow.tmp.gather(
            flow.Tensor(input), flow.Tensor(index, dtype=flow.int), dim=2
        )
        test_case.assertTrue(np.allclose(output2.numpy(), np_out2))

        np_out3 = gather_numpy(input, index, dim=3)
        output3 = flow.tmp.gather(
            flow.Tensor(input), flow.Tensor(index, dtype=flow.int), dim=3
        )
        test_case.assertTrue(np.allclose(output3.numpy(), np_out3))


if __name__ == "__main__":
    unittest.main()
