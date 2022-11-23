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
import random as random_util
import unittest

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *
from oneflow.nn.common_types import _size_1_t, _size_2_t, _size_3_t

# y = pool(x), z = unpool(y, indices), pool_input_shape is x.shape, pool_output_shape is y.shape.
# When `output_size` in unpool() is empty, the op will calculate the output size according to
# kernel_size, stride and padding. But when index in indices is outside the range required
# by output_size calculated by unpool op, the value of result and related grad will be unknown.
# To avoid the problem, this function calculate the output_size which will not cause unknown problems.
def _get_valid_output_size(
    pool_input_shape, pool_output_shape, kernel_size, stride, padding
):
    def convert_data(data, i, dst_data=None):
        if not isinstance(data, (list, int)):
            return dst_data
        if isinstance(data, list):
            return data[i]
        return data

    _, _, *pool_input_hwd_shape = pool_input_shape.pytorch
    batch_size, num_channels, *pool_out_hwd_shape = pool_output_shape.pytorch
    unpool_output_shape = [batch_size, num_channels]
    for i, (pool_input_size, pool_output_size) in enumerate(
        zip(pool_input_hwd_shape, pool_out_hwd_shape)
    ):
        kernel_size_value = convert_data(kernel_size.value(), i)
        stride_value = convert_data(stride.value(), i, kernel_size_value)
        padding_value = convert_data(padding.value(), i, 0)
        unpool_output_size = max(
            pool_input_size,
            (pool_output_size - 1) * stride_value
            - 2 * padding_value
            + kernel_size_value,
        )
        unpool_output_shape.append(unpool_output_size)
    return torch.Size(unpool_output_shape)


def _test_module_unpoolnd(test_case, n):
    device = random_device()
    if n == 1:
        _size_n_t = _size_1_t
        MaxPoolNd = torch.nn.MaxPool1d
        MaxUnpoolNd = torch.nn.MaxUnpool1d
        x = random_tensor(ndim=3, dim2=random(20, 31), requires_grad=False).to(device)
    elif n == 2:
        _size_n_t = _size_2_t
        MaxPoolNd = torch.nn.MaxPool2d
        MaxUnpoolNd = torch.nn.MaxUnpool2d
        x = random_tensor(
            ndim=4, dim2=random(20, 31), dim3=random(20, 31), requires_grad=False
        ).to(device)
    elif n == 3:
        _size_n_t = _size_3_t
        MaxPoolNd = torch.nn.MaxPool3d
        MaxUnpoolNd = torch.nn.MaxUnpool3d
        x = random_tensor(
            ndim=5,
            dim2=random(20, 31),
            dim3=random(20, 31),
            dim4=random(20, 31),
            requires_grad=False,
        ).to(device)

    kernel_size = random(4, 6).to(_size_n_t)
    stride = random(1, 3).to(_size_n_t) | nothing()
    padding = random(1, 3).to(_size_n_t) | nothing()
    m = MaxPoolNd(
        kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True,
    )
    m.train(random())
    m.to(device)
    y = m(x)
    pooling_results_dtype = random_util.choice(
        [torch.int, torch.long, torch.float, torch.double]
    )
    indices_dtype = random_util.choice([torch.int, torch.long])
    pooling_results = y[0].to(pooling_results_dtype)
    indices = y[1].to(indices_dtype)
    pooling_results.requires_grad_()
    output_size = _get_valid_output_size(
        x.shape, pooling_results.shape, kernel_size, stride, padding
    )
    unpool_module = MaxUnpoolNd(
        kernel_size=kernel_size, stride=stride, padding=padding,
    )
    result = unpool_module(pooling_results, indices, output_size=output_size)
    return result


def _test_functional_unpoolnd(test_case, n):
    device = random_device()

    if n == 1:
        _size_n_t = _size_1_t
        MaxPoolNd = torch.nn.MaxPool1d
        max_unpool_nd = torch.nn.functional.max_unpool1d
        x = random_tensor(ndim=3, dim2=random(20, 31), requires_grad=False).to(device)
    elif n == 2:
        _size_n_t = _size_2_t
        MaxPoolNd = torch.nn.MaxPool2d
        max_unpool_nd = torch.nn.functional.max_unpool2d
        x = random_tensor(
            ndim=4, dim2=random(20, 31), dim3=random(20, 31), requires_grad=False
        ).to(device)
    elif n == 3:
        _size_n_t = _size_3_t
        MaxPoolNd = torch.nn.MaxPool3d
        max_unpool_nd = torch.nn.functional.max_unpool3d
        x = random_tensor(
            ndim=5,
            dim2=random(20, 31),
            dim3=random(20, 31),
            dim4=random(20, 31),
            requires_grad=False,
        ).to(device)

    kernel_size = random(4, 6).to(_size_n_t)
    stride = random(1, 3).to(_size_n_t) | nothing()
    padding = random(1, 3).to(_size_n_t) | nothing()
    m = MaxPoolNd(
        kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True,
    )
    m.train(random())
    m.to(device)
    y = m(x)
    pooling_results_dtype = random_util.choice(
        [torch.int, torch.long, torch.float, torch.double]
    )
    indices_dtype = random_util.choice([torch.int, torch.long])
    pooling_results = y[0].to(pooling_results_dtype)
    indices = y[1].to(indices_dtype)
    pooling_results.requires_grad_()
    output_size = _get_valid_output_size(
        x.shape, pooling_results.shape, kernel_size, stride, padding
    )
    return max_unpool_nd(
        pooling_results,
        indices,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_size=output_size,
    )


@flow.unittest.skip_unless_1n1d()
class TestMaxUnpooling(flow.unittest.TestCase):
    @autotest(n=3, check_graph=False)
    def test_max_unpool1d_with_random_data(test_case):
        return _test_module_unpoolnd(test_case, 1)

    @autotest(n=3, check_graph=False)
    def test_functional_max_unpool1d_with_random_data(test_case):
        return _test_functional_unpoolnd(test_case, 1)

    @autotest(n=3, check_graph=False)
    def test_max_unpool2d_with_random_data(test_case):
        return _test_module_unpoolnd(test_case, 2)

    @autotest(n=3, check_graph=False)
    def test_functional_max_unpool2d_with_random_data(test_case):
        return _test_functional_unpoolnd(test_case, 2)

    @autotest(n=3, check_graph=False)
    def test_max_unpool3d_with_random_data(test_case):
        return _test_module_unpoolnd(test_case, 3)

    @autotest(n=3, check_graph=False)
    def test_functional_max_unpool3d_with_random_data(test_case):
        return _test_functional_unpoolnd(test_case, 3)

    @profile(torch.nn.functional.max_unpool1d)
    def profile_max_unpool1d(test_case):
        max_pool_results = torch.randn(1, 32, 64)
        max_pool_indices = torch.arange(64).expand(1, 32, 64)
        torch.nn.functional.max_unpool1d(max_pool_results, max_pool_indices, 2)

        max_pool_results = torch.randn(32, 32, 64)
        max_pool_indices = torch.arange(64).expand(32, 32, 64)
        torch.nn.functional.max_unpool1d(max_pool_results, max_pool_indices, 2)

    @profile(torch.nn.functional.max_unpool2d)
    def profile_max_unpool2d(test_case):
        max_pool_results = torch.randn(1, 16, 32, 32)
        max_pool_indices = torch.arange(32).expand(1, 16, 32, 32)
        torch.nn.functional.max_unpool2d(max_pool_results, max_pool_indices, 2)

        max_pool_results = torch.randn(32, 16, 32, 32)
        max_pool_indices = torch.arange(32).expand(32, 16, 32, 32)
        torch.nn.functional.max_unpool2d(max_pool_results, max_pool_indices, 2)

    @profile(torch.nn.functional.max_unpool3d)
    def profile_max_unpool3d(test_case):
        max_pool_results = torch.randn(1, 4, 32, 32, 32)
        max_pool_indices = torch.arange(32).expand(1, 4, 32, 32, 32)
        torch.nn.functional.max_unpool3d(max_pool_results, max_pool_indices, 2)

        max_pool_results = torch.randn(16, 4, 32, 32, 32)
        max_pool_indices = torch.arange(32).expand(16, 4, 32, 32, 32)
        torch.nn.functional.max_unpool3d(max_pool_results, max_pool_indices, 2)


if __name__ == "__main__":
    unittest.main()
