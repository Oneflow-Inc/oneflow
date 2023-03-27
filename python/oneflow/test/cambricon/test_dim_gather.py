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
import itertools
import numpy as np
import oneflow as flow


def __test_dim_gather(x_array, index_array, dim, index_dtype):
    x = flow.tensor(x_array, device="cpu", dtype=flow.float32)
    index = flow.tensor(index_array, device="cpu", dtype=index_dtype)
    cpu_out_numpy = flow.gather(x, dim, index).numpy()
    x = x.to("mlu")
    index = index.to("mlu")
    mlu_out_numpy = flow.gather(x, dim, index).numpy()
    assert np.allclose(cpu_out_numpy, mlu_out_numpy, 1e-4, 1e-4)


def __test_dim_gather_backward(x_array, index_array, dim, index_dtype):
    x_cpu = flow.tensor(x_array, device="cpu", dtype=flow.float32, requires_grad=True)
    index = flow.tensor(index_array, device="cpu", dtype=index_dtype)
    out_cpu = flow.gather(x_cpu, dim, index)
    out_cpu.sum().backward()

    x_mlu = flow.tensor(x_array, device="mlu", dtype=flow.float32, requires_grad=True)
    index = index.to("mlu")
    out_mlu = flow.gather(x_mlu, dim, index)
    out_mlu.sum().backward()
    assert np.allclose(out_cpu.numpy(), out_mlu.cpu().numpy(), 1e-4, 1e-4)
    assert np.allclose(x_cpu.grad.numpy(), x_mlu.grad.cpu().numpy(), 1e-4, 1e-4)


def test_dim_gather():
    array = (
        np.array([[1, 2], [3, 4]]),
        np.array([[0, 0], [1, 0]]),
    )
    array_multi_dim = (
        np.random.randn(3, 4, 3, 5),
        np.random.choice(np.arange(3), size=180, replace=True).reshape((3, 4, 3, 5)),
    )
    arrays_single_dim = zip([np.ones(1), 1.0], [0, 0])

    index_dtypes = [flow.int32, flow.int64]

    for dim, dtype in itertools.product([0, 1], index_dtypes):
        __test_dim_gather(*array, dim, dtype)
        __test_dim_gather_backward(*array, dim, dtype)

    for dim, dtype in itertools.product([1, 2, 3], index_dtypes):
        __test_dim_gather(*array_multi_dim, dim, dtype)
        __test_dim_gather_backward(*array_multi_dim, dim, dtype)

    for array_pair, dtype in itertools.product(arrays_single_dim, index_dtypes):
        __test_dim_gather(*array_pair, 0, dtype)
        __test_dim_gather_backward(*array_pair, 0, dtype)
