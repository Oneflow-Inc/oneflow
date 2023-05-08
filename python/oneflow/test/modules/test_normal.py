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
from collections import OrderedDict

from oneflow.test_utils.test_util import GenArgList, type_name_to_flow_type
from oneflow.test_utils.automated_test_util import *
import oneflow as flow


def _test_normal(test_case, mean, std, shape, device, dtype):
    dtype = type_name_to_flow_type[dtype]
    device = flow.device(device)
    y1 = flow.normal(mean, std, shape, dtype=dtype, device=device)
    y2 = flow.normal(mean, std, size=shape, dtype=dtype, device=device)
    test_case.assertFalse(np.array_equal(y1.numpy(), y2.numpy()))
    test_case.assertEqual(shape, y1.shape)
    test_case.assertEqual(dtype, y1.dtype)
    test_case.assertEqual(shape, y2.shape)
    test_case.assertEqual(dtype, y2.dtype)

    # NOTE(Feng Wen): The test code  helper is modified from  def test_normal(self, device, dtype):
    # https://github.com/pytorch/pytorch/blob/e63c502baa4a6f2109749984be701e722b3b7232/test/test_tensor_creation_ops.py#L3073-L3219
    def helper(self, device, dtype, ptype, t_transform, std_transform):
        q = flow.empty(100, 100, dtype=dtype, device=device)

        q.normal_()
        self.assertTrue(np.allclose(t_transform(q).mean().item(), 0, atol=0.2, rtol=0))
        self.assertTrue(
            np.allclose(t_transform(q).std().item(), std_transform(1), atol=0.2, rtol=0)
        )

        q.normal_(2, 3)
        self.assertTrue(np.allclose(t_transform(q).mean().item(), 2, atol=0.3, rtol=0))
        self.assertTrue(
            np.allclose(t_transform(q).std().item(), std_transform(3), atol=0.3, rtol=0)
        )

        q = flow.empty(100, 100, dtype=dtype, device=device)
        q_row1 = q[0:1].clone()
        q[99:100].normal_()
        self.assertTrue(
            np.allclose(t_transform(q[99:100]).mean().item(), 0, atol=0.4, rtol=0)
        )
        self.assertTrue(
            np.allclose(
                t_transform(q[99:100]).std().item(), std_transform(1), atol=0.3, rtol=0
            )
        )
        self.assertTrue(flow.allclose(t_transform(q[0:1]).clone(), t_transform(q_row1)))

        mean = flow.empty(100, 100, dtype=dtype, device=device)
        mean[:50].fill_(ptype(0))
        mean[50:].fill_(ptype(1))

        std = flow.empty(100, 100, dtype=flow.float, device=device)
        std[:, :50] = 4
        std[:, 50:] = 1

        r = flow.normal(mean)
        self.assertEqual(r.dtype, dtype)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(
            np.allclose(t_transform(r[:50]).mean().item(), 0, atol=0.2, rtol=0)
        )
        self.assertTrue(np.allclose(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0))
        self.assertTrue(
            np.allclose(t_transform(r).std(), std_transform(1), atol=0.2, rtol=0)
        )

        r.fill_(42)
        r = flow.normal(mean, 3)
        self.assertEqual(r.dtype, dtype)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(np.allclose(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0))
        self.assertTrue(np.allclose(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0))
        self.assertTrue(
            np.allclose(t_transform(r).std(), std_transform(3), atol=0.2, rtol=0)
        )

        r.fill_(42)
        flow.normal(mean, 3, out=r)
        self.assertEqual(r.dtype, dtype)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(np.allclose(t_transform(r[:50]).mean(), 0, atol=0.2, rtol=0))
        self.assertTrue(np.allclose(t_transform(r[50:]).mean(), 1, atol=0.2, rtol=0))
        self.assertTrue(
            np.allclose(t_transform(r).std(), std_transform(3), atol=0.2, rtol=0)
        )

        r.fill_(42)
        r = flow.normal(2, std)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(np.allclose(r.mean().numpy(), 2, atol=0.2, rtol=0))
        self.assertTrue(np.allclose(r[:, :50].std().numpy(), 4, atol=0.3, rtol=0))
        self.assertTrue(np.allclose(r[:, 50:].std().numpy(), 1, atol=0.2, rtol=0))

        r.fill_(42)
        flow.normal(2, std, out=r)
        self.assertFalse(r.dtype.is_complex)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(np.allclose(r.mean().numpy(), 2, atol=0.2, rtol=0))
        self.assertTrue(np.allclose(r[:, :50].std().numpy(), 4, atol=0.3, rtol=0))
        self.assertTrue(np.allclose(r[:, 50:].std().numpy(), 1, atol=0.2, rtol=0))

        r.fill_(42)
        r = flow.normal(mean, std)
        self.assertEqual(r.dtype, dtype)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(
            np.allclose(t_transform(r[:50]).mean().numpy(), 0, atol=0.2, rtol=0)
        )
        self.assertTrue(
            np.allclose(t_transform(r[50:]).mean().numpy(), 1, atol=0.2, rtol=0)
        )
        self.assertTrue(
            np.allclose(
                t_transform(r[:, :50]).std().numpy(), std_transform(4), atol=0.3, rtol=0
            )
        )
        self.assertTrue(
            np.allclose(
                t_transform(r[:, 50:]).std().numpy(), std_transform(1), atol=0.2, rtol=0
            )
        )

        r.fill_(42)
        flow.normal(mean, std, out=r)
        self.assertEqual(r.dtype, dtype)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(
            np.allclose(t_transform(r[:50]).mean().numpy(), 0, atol=0.2, rtol=0)
        )
        self.assertTrue(
            np.allclose(t_transform(r[50:]).mean().numpy(), 1, atol=0.2, rtol=0)
        )
        self.assertTrue(
            np.allclose(
                t_transform(r[:, :50]).std().numpy(), std_transform(4), atol=0.3, rtol=0
            )
        )
        self.assertTrue(
            np.allclose(
                t_transform(r[:, 50:]).std().numpy(), std_transform(1), atol=0.2, rtol=0
            )
        )

        # test empty mean/std
        out = flow.normal(mean=flow.empty((0, 2)), std=flow.empty((0, 1)))
        self.assertEqual(out.size(), flow.Size([0, 2]))

        r.fill_(42)
        r = flow.normal(2, 3, (100, 100), dtype=dtype, device=device)
        self.assertEqual(r.dtype, dtype)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(np.allclose(t_transform(r).mean().numpy(), 2, atol=0.3, rtol=0))
        self.assertTrue(
            np.allclose(
                t_transform(r).std().numpy(), std_transform(3), atol=0.3, rtol=0
            )
        )

        r.fill_(42)
        flow.normal(2, 3, (100, 100), dtype=dtype, device=device, out=r)
        self.assertEqual(r.dtype, dtype)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(np.allclose(t_transform(r).mean().numpy(), 2, atol=0.3, rtol=0))
        self.assertTrue(
            np.allclose(
                t_transform(r).std().numpy(), std_transform(3), atol=0.3, rtol=0
            )
        )

        # float std 0 with float mean
        r.fill_(42)
        r = flow.normal(2, 0, (10, 10), dtype=dtype, device=device)
        self.assertEqual(r.dtype, dtype)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(np.allclose(r.numpy(), 2))

        # float std 0 with tensor mean
        r.fill_(42)
        mean_rand = flow.randn(10, 10, dtype=dtype, device=device)
        flow.normal(mean_rand, 0, out=r)
        self.assertEqual(r.dtype, dtype)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(np.allclose(mean_rand.numpy(), r.numpy(), atol=0, rtol=0))

        # tensor std 0 with float mean
        r.fill_(42)
        std_zeros = flow.zeros(10, 10, dtype=dtype, device=device)
        flow.normal(2, std_zeros, out=r)
        self.assertEqual(r.dtype, dtype)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(np.allclose(r.numpy(), 2))

        # tensor std 0 with tensor mean
        r.fill_(42)
        flow.normal(mean_rand, std_zeros, out=r)
        self.assertEqual(r.dtype, dtype)
        self.assertEqual(str(r.device), str(device))
        self.assertTrue(np.allclose(mean_rand.numpy(), r.numpy(), atol=0, rtol=0))

    helper(test_case, device, dtype, lambda x: x, lambda t: t, lambda mean: mean)


def _test_with_generator(test_case, mean, std, shape, device, dtype):
    dtype = type_name_to_flow_type[dtype]
    gen = flow.Generator()
    gen.manual_seed(0)
    y1 = flow.normal(
        mean, std, shape, generator=gen, dtype=dtype, device=flow.device(device)
    )
    gen.manual_seed(0)
    y2 = flow.normal(
        mean, std, shape, generator=gen, dtype=dtype, device=flow.device(device)
    )
    test_case.assertTrue(np.array_equal(y1.numpy(), y2.numpy()))


def _test_backward(test_case, mean, std, shape, device, dtype):
    dtype = type_name_to_flow_type[dtype]
    x = flow.normal(
        mean, std, shape, dtype=dtype, device=flow.device(device), requires_grad=True
    )
    y = x.sum()
    y.backward()
    test_case.assertTrue(np.array_equal(np.ones(shape), x.grad.numpy()))


@flow.unittest.skip_unless_1n1d()
class TestNormModule(flow.unittest.TestCase):
    def test_norm(test_case):
        arg_dict = OrderedDict()
        arg_dict["fun"] = [_test_normal, _test_with_generator, _test_backward]
        arg_dict["mean"] = [-1, 0, 1]
        arg_dict["std"] = [1, 2, 8]
        arg_dict["shape"] = [(2, 3), (2, 3, 4), (2, 3, 4, 5)]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = ["float32", "double"]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
