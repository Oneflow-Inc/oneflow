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
import os
import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest


def _to_last(ndim):
    return [0] + list(range(2, ndim)) + [1]


def _to_first(ndim):
    return [0, ndim - 1] + list(range(1, ndim - 1))


def _check_against_channels_first(test_case, pool, shape, device, dtype):
    # Distinct values, so max pooling has no ties to break differently.
    n = int(np.prod(shape))
    x_np = (np.random.permutation(n).reshape(shape) / n).astype(dtype)
    ndim = len(shape)

    x_cf = flow.tensor(x_np, device=device, requires_grad=True)
    y_cf = pool(x_cf, "channels_first")
    x_cl = flow.tensor(
        np.ascontiguousarray(x_np.transpose(_to_last(ndim))),
        device=device,
        requires_grad=True,
    )
    y_cl = pool(x_cl, "channels_last")
    test_case.assertTrue(
        np.array_equal(y_cl.numpy().transpose(_to_first(ndim)), y_cf.numpy())
    )

    dy = np.random.uniform(-1, 1, size=y_cf.shape).astype(dtype)
    (y_cf * flow.tensor(dy, device=device)).sum().backward()
    dy_cl = np.ascontiguousarray(dy.transpose(_to_last(ndim)))
    (y_cl * flow.tensor(dy_cl, device=device)).sum().backward()
    test_case.assertTrue(
        np.allclose(
            x_cl.grad.numpy().transpose(_to_first(ndim)),
            x_cf.grad.numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
    )


def _max_pool2d(dilation, return_indices):
    def pool(x, data_format):
        return flow._C.max_pool2d(
            x,
            kernel_size=[3, 2],
            stride=[2, 1],
            padding=[1, 1],
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=False,
            data_format=data_format,
        )[0]

    return pool


def _devices():
    if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
        return ["cpu"]
    return ["cpu", "cuda"]


@flow.unittest.skip_unless_1n1d()
class TestPoolingChannelsLast(flow.unittest.TestCase):
    def test_max_pool2d_native_channels_last(test_case):
        for device in _devices():
            for dilation in ([1, 1], [2, 2]):
                for return_indices in (False, True):
                    for dtype in (np.float32, np.float64):
                        _check_against_channels_first(
                            test_case,
                            _max_pool2d(dilation, return_indices),
                            (2, 3, 9, 8),
                            device,
                            dtype,
                        )


if __name__ == "__main__":
    unittest.main()
