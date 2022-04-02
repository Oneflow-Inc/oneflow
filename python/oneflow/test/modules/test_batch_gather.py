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
from collections import OrderedDict
import os

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest


def _test_batch_gather(test_case, shape, device):
    # for example: shape = (3, 2, 2)
    x = np.random.randn(*shape)
    x_tensor = flow.Tensor(x).to(device)
    x_tensor.requires_grad = True
    batchsize = x.shape[0]
    init_index = np.array(
        [np.random.randint(batchsize) for i in range(batchsize)]
    ).astype(np.int64)

    batch_gather_index = flow.tensor(init_index).to(device)
    batch_gather_out = flow.batch_gather(x_tensor, batch_gather_index)

    x_tensor_gather = flow.Tensor(x).to(device)
    x_tensor_gather.requires_grad = True
    reshaped_shape = [batchsize]  # reshaped_shape = [3]
    for i in range(len(x.shape) - 1):
        reshaped_shape.append(1)  # reshaped_shape = [3] -> [3, 1, 1]

    gather_index = np.reshape(init_index, reshaped_shape)
    gather_index = np.broadcast_to(gather_index, shape).astype(
        np.int64
    )  # [3, 1, 1] -> [3, 2, 2]
    gather_index = flow.tensor(gather_index).to(device)
    gather_out = flow.gather(x_tensor_gather, 0, gather_index)
    total_out = batch_gather_out.sum() + gather_out.sum()
    total_out.backward()

    test_case.assertTrue(
        np.allclose(batch_gather_out.numpy(), gather_out.numpy(), atol=1e-4, rtol=1e-4)
    )

    test_case.assertTrue(
        np.allclose(
            x_tensor.grad.numpy(), x_tensor_gather.grad.numpy(), atol=1e-4, rtol=1e-4,
        )
    )
    test_case.assertTrue(
        np.allclose(
            x_tensor.grad.numpy(), x_tensor_gather.grad.numpy(), atol=1e-4, rtol=1e-4,
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestBatchGather(flow.unittest.TestCase):
    def test_batch_gather(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_batch_gather]
        arg_dict["shape"] = [(3, 2, 2), (3, 2, 4, 2), (3, 3, 4, 2, 2), (4, 2)]
        arg_dict["device"] = ["cpu", "cuda"]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
