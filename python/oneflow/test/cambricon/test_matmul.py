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

import oneflow as flow
import oneflow.unittest


def _test_matmul_forward(test_case, shape, device, dtype):
    (shape_a, shape_b, transpose_a, transpose_b) = shape
    alpha = np.random.randn()
    a = np.random.randn(*shape_a)
    b = np.random.randn(*shape_b)

    # mlu
    mlu_a = flow.tensor(a, device=flow.device(device), dtype=dtype)
    mlu_b = flow.tensor(b, device=flow.device(device), dtype=dtype)
    mlu_out = flow.matmul(
        mlu_a, mlu_b, alpha=alpha, transpose_a=transpose_a, transpose_b=transpose_b
    )
    # cpu
    cpu_a = flow.tensor(a, device=flow.device("cpu"), dtype=dtype)
    cpu_b = flow.tensor(b, device=flow.device("cpu"), dtype=dtype)
    cpu_out = flow.matmul(
        cpu_a, cpu_b, alpha=alpha, transpose_a=transpose_a, transpose_b=transpose_b
    )
    # compare
    diff = 0.0001
    test_case.assertTrue(np.allclose(mlu_out.numpy(), cpu_out.numpy(), diff, diff))


@flow.unittest.skip_unless_1n1d()
class TestBatchMatmulCambriconModule(flow.unittest.TestCase):
    def test_broadcast_matmul_grad_b_grad(test_case):
        broadcast_dims = [
            np.random.randint(2, 10) for _ in range(np.random.randint(1, 3))
        ]
        m = np.random.randint(2, 10)
        n = np.random.randint(2, 10)
        k = np.random.randint(2, 10)

        shape_a = broadcast_dims + [m, k]
        shape_b = [k, n]
        shape_y = broadcast_dims + [m, n]
        dtype = flow.float32

        arr_a = np.random.randn(*shape_a)
        arr_b = np.random.randn(*shape_b)
        arr_grad_y = np.random.randn(*shape_y)

        a = flow.tensor(arr_a, device="cpu", dtype=dtype).requires_grad_(True)
        b = flow.tensor(arr_b, device="cpu", dtype=dtype).requires_grad_(True)
        y = flow.matmul(a, b)

        a_mlu = flow.tensor(arr_a, device="mlu", dtype=dtype).requires_grad_(True)
        b_mlu = flow.tensor(arr_b, device="mlu", dtype=dtype).requires_grad_(True)
        y_mlu = flow.matmul(a_mlu, b_mlu)

        init_grad_y = flow.tensor(arr_grad_y, device="cpu", dtype=dtype).requires_grad_(
            True
        )
        init_grad_y_mlu = flow.tensor(
            arr_grad_y, device="mlu", dtype=dtype
        ).requires_grad_(True)

        da = flow.autograd.grad(
            outputs=y,
            inputs=a,
            grad_outputs=init_grad_y,
            create_graph=True,
            retain_graph=True,
        )[0]
        da_mlu = flow.autograd.grad(
            outputs=y_mlu,
            inputs=a_mlu,
            grad_outputs=init_grad_y_mlu,
            create_graph=True,
            retain_graph=True,
        )[0]
        test_case.assertTrue(
            np.allclose(
                da.detach().cpu().numpy(),
                da_mlu.detach().cpu().numpy(),
                rtol=1e-4,
                atol=1e-4,
            )
        )

        db = flow.autograd.grad(
            outputs=y,
            inputs=b,
            grad_outputs=init_grad_y,
            create_graph=True,
            retain_graph=True,
        )[0]
        db_mlu = flow.autograd.grad(
            outputs=y_mlu,
            inputs=b_mlu,
            grad_outputs=init_grad_y_mlu,
            create_graph=True,
            retain_graph=True,
        )[0]
        test_case.assertTrue(
            np.allclose(
                db.detach().cpu().numpy(),
                db_mlu.detach().cpu().numpy(),
                rtol=1e-4,
                atol=1e-4,
            )
        )

    def test_matmul(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_matmul_forward,
        ]
        arg_dict["shape"] = [
            ((1, 1), (1, 1), False, False),
            ((1, 3), (3, 1), False, False),
            ((1, 3), (3, 1), True, True),
            ((2, 3), (3, 4), False, False),
            ((2, 3), (4, 3), False, True),
            ((3, 2), (3, 4), True, False),
            ((3, 2), (4, 3), True, True),
        ]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_batch_matmul(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_matmul_forward,
        ]
        arg_dict["shape"] = [
            ((2, 3, 4), (2, 4, 5), False, False,),
            ((2, 4, 5), (2, 5, 6), False, False,),
            ((2, 3, 4, 5), (2, 3, 5, 6), False, False,),
            ((2, 4, 3), (2, 4, 5), True, False,),
            ((2, 4, 5), (2, 6, 5), False, True,),
            ((2, 3, 5, 4), (2, 3, 6, 5), True, True,),
        ]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_broadcast_matmul(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            _test_matmul_forward,
        ]
        arg_dict["shape"] = [
            ((1, 3, 4), (2, 4, 5), False, False,),
            ((1, 7, 3, 4), (7, 1, 4, 5), False, False,),
            ((1, 4, 3), (2, 4, 5), True, False,),
            ((1, 7, 3, 4), (7, 1, 5, 4), False, True,),
        ]
        arg_dict["device"] = ["mlu"]
        arg_dict["dtype"] = [
            flow.float,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
