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

import numpy as np
from oneflow.test_utils.test_util import GenArgList
from oneflow.test_utils.automated_test_util import *

import oneflow as flow
import oneflow.unittest


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


def _test_gather(test_case, device):
    input = np.array([[1, 2], [3, 4]])
    index = np.array([[0, 0], [1, 0]])
    np_out = np.take_along_axis(input, index, 0)
    output = flow.gather(
        flow.tensor(input, dtype=flow.float32, device=flow.device(device)),
        0,
        flow.tensor(index, dtype=flow.int64, device=flow.device(device)),
    )
    test_case.assertTrue(np.array_equal(output.numpy(), np_out))


def _test_gather_tensor_function(test_case, device):
    input = np.array([[1, 2], [3, 4]])
    index = np.array([[0, 0], [1, 0]])
    np_out = np.take_along_axis(input, index, 1)
    input = flow.tensor(input, dtype=flow.float32, device=flow.device(device))
    index = flow.tensor(index, dtype=flow.int64, device=flow.device(device))
    output = input.gather(1, index)
    test_case.assertTrue(np.array_equal(output.numpy(), np_out))


def _test_gather_random_array(test_case, device):
    input = np.random.randn(3, 4, 3, 5)
    index = np.random.choice(np.arange(3), size=180, replace=True).reshape((3, 4, 3, 5))
    np_out = np.take_along_axis(input, index, 1)
    output = flow.gather(
        flow.tensor(input, dtype=flow.float32, device=flow.device(device)),
        1,
        flow.tensor(index, dtype=flow.int64, device=flow.device(device)),
    )
    test_case.assertTrue(np.allclose(output.numpy(), np_out))
    np_out2 = np.take_along_axis(input, index, 2)
    output2 = flow.gather(
        flow.tensor(input, dtype=flow.float32, device=flow.device(device)),
        2,
        flow.tensor(index, dtype=flow.int64, device=flow.device(device)),
    )
    test_case.assertTrue(np.allclose(output2.numpy(), np_out2))
    np_out3 = np.take_along_axis(input, index, 3)
    output3 = flow.gather(
        flow.tensor(input, dtype=flow.float32, device=flow.device(device)),
        3,
        flow.tensor(index, dtype=flow.int64, device=flow.device(device)),
    )
    test_case.assertTrue(np.allclose(output3.numpy(), np_out3))


def _test_gather_backward(test_case, device):
    input = np.array([[1, 2], [3, 4]])
    index = np.array([[0, 0], [1, 0]])
    np_out = np.take_along_axis(input, index, 0)
    np_grad = _scatter_add_numpy(np.ones_like(np_out), 0, index, input.shape)
    of_input = flow.tensor(
        input, dtype=flow.float32, requires_grad=True, device=flow.device(device)
    )
    output = flow.gather(
        of_input, 0, flow.tensor(index, dtype=flow.int64, device=flow.device(device)),
    )
    out_sum = output.sum()
    out_sum.backward()
    test_case.assertTrue(np.array_equal(output.numpy(), np_out))
    test_case.assertTrue(np.array_equal(of_input.grad.numpy(), np_grad))


def _test_gather_index_0dim_tensor(test_case, device):
    input = flow.ones(1).to(device)
    input.requires_grad = True
    index = flow.tensor(0).to(device)
    output = flow.gather(input, 0, index)
    test_case.assertTrue(np.array_equal(output.numpy(), 1.0))
    output.sum().backward()
    test_case.assertTrue(np.array_equal(input.grad.numpy(), [1.0]))


def _test_gather_input_index_0dim_tensor(test_case, device):
    input = flow.tensor(1.0).to(device)
    input.requires_grad = True
    index = flow.tensor(0).to(device)
    output = flow.gather(input, 0, index)
    test_case.assertTrue(np.array_equal(output.numpy(), 1.0))
    output.sum().backward()
    test_case.assertTrue(np.array_equal(input.grad.numpy(), 1.0))


def _test_gather_input_0dim_tensor(test_case, device):
    input = flow.tensor(1.0).to(device)
    input.requires_grad = True
    index = flow.tensor([0]).to(device)
    output = flow.gather(input, 0, index)
    test_case.assertTrue(np.array_equal(output.numpy(), [1.0]))
    output.sum().backward()
    test_case.assertTrue(np.array_equal(input.grad.numpy(), 1.0))


@flow.unittest.skip_unless_1n1d()
class TestGather(flow.unittest.TestCase):
    def test_gather(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_gather,
            _test_gather_tensor_function,
            _test_gather_random_array,
            _test_gather_backward,
            _test_gather_index_0dim_tensor,
            _test_gather_input_index_0dim_tensor,
            _test_gather_input_0dim_tensor,
        ]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @autotest(n=5)
    def test_flow_gather_with_random_data(test_case):
        device = random_device()
        input = random_tensor(ndim=4, dim0=3, dim1=3, dim2=4, dim3=5).to(device)
        dim = random(-4, 4).to(int)
        index = random_tensor(
            ndim=4,
            dim1=random(1, 3).to(int),
            dim2=random(1, 4).to(int),
            dim3=random(1, 5).to(int),
            low=0,
            high=3,
            dtype=int,
        ).to(device)
        return torch.gather(input, dim, index)

    @autotest(n=5, auto_backward=False, check_graph=True)
    def test_flow_gather_bool_with_random_data(test_case):
        device = random_device()
        input = random_tensor(ndim=4, dim0=3, dim1=3, dim2=4, dim3=5).to(
            device=device, dtype=torch.bool
        )
        dim = random(0, 4).to(int)
        index = random_tensor(
            ndim=4,
            dim1=random(1, 3).to(int),
            dim2=random(1, 4).to(int),
            dim3=random(1, 5).to(int),
            low=0,
            high=3,
            dtype=int,
        ).to(device)
        return torch.gather(input, dim, index)

    @profile(torch.gather)
    def profile_gather(test_case):
        t = torch.ones(1000, 1000)
        torch.gather(t, 1, torch.ones(1000, 1000, dtype=torch.int64))


if __name__ == "__main__":
    unittest.main()
