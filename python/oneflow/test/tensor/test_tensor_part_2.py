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

import copy
import os
import unittest
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestTensor(flow.unittest.TestCase):
    @autotest(n=10)
    def test_permute_flow_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        permute_list = [0, 1, 2, 3]
        np.random.shuffle(permute_list)
        y = x.permute(permute_list)
        return y

    @autotest(n=1)
    def test_permute_flow_with_random_data_and_keyword(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        permute_list = [0, 1, 2, 3]
        np.random.shuffle(permute_list)
        y = x.permute(dims=permute_list)
        return y

    @autotest(n=5)
    def test_transpose_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        permute_list = np.random.permutation(4)
        y = x.transpose(permute_list[0], permute_list[1])
        return y

    @autotest(n=5)
    def test_t_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(
            ndim=constant(2).to(int), dim0=random(0, 64), dim1=random(0, 64)
        ).to(device)
        y = x.t()
        return y

    @autotest(n=5)
    def test_T_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=random(1, 4)).to(device)
        y = x.T
        return y

    def test_tensor_where(test_case):
        x = flow.tensor(
            np.array([[-0.462, 0.3139], [0.3898, -0.7197], [0.0478, -0.1657]]),
            dtype=flow.float32,
        )
        y = flow.tensor(np.ones(shape=(3, 2)), dtype=flow.float32)
        condition = flow.tensor(np.array([[0, 1], [1, 0], [1, 0]]), dtype=flow.int32)
        of_out = condition.where(x, y)
        np_out = np.array([[1.0, 0.3139], [0.3898, 1.0], [0.0478, 1.0]])
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_tensor_equal(test_case):
        arr1 = np.random.randint(1, 10, size=(2, 3, 4, 5))
        arr2 = np.random.randint(1, 10, size=(2, 3, 4, 5))
        input = flow.tensor(arr1, dtype=flow.float32)
        other = flow.tensor(arr2, dtype=flow.float32)
        of_out = input.eq(other)
        np_out = np.equal(arr1, arr2)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_tensor_equal_bool_dtype(test_case):
        np_bool = np.random.randint(0, 2, size=()).astype(bool).item()
        input = flow.tensor(np_bool, dtype=flow.bool)
        input2 = flow.tensor([np_bool], dtype=flow.bool)
        test_case.assertTrue(input == np_bool)
        test_case.assertTrue(input2 == np_bool)

    def test_tensor_detach(test_case):
        shape = (2, 3, 4, 5)
        x = flow.tensor(np.random.randn(*shape), dtype=flow.float32, requires_grad=True)
        test_case.assertTrue(np.allclose(x.detach().numpy(), x.numpy(), 0.0001, 0.0001))
        test_case.assertEqual(x.detach().requires_grad, False)
        y = x * 2
        z = y.detach()
        test_case.assertEqual(z.is_leaf, True)
        test_case.assertEqual(z.grad_fn, None)

    def _test_cast_tensor_function(test_case):
        shape = (2, 3, 4, 5)
        np_arr = np.random.randn(*shape).astype(np.float32)
        input = flow.tensor(np_arr, dtype=flow.float32)
        output = input.cast(flow.int8)
        np_out = np_arr.astype(np.int8)
        test_case.assertTrue(np.allclose(output.numpy(), np_out))

    def _test_sin_tensor_function(test_case, shape, device):
        input = flow.Tensor(np.random.randn(2, 3, 4, 5))
        of_out = input.sin()
        np_out = np.sin(input.numpy())
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))

    def test_cos_tensor_function(test_case):
        arr = np.random.randn(2, 3, 4, 5)
        input = flow.tensor(arr, dtype=flow.float32)
        np_out = np.cos(arr)
        of_out = input.cos()
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))

    def test_std_tensor_function(test_case):
        np_arr = np.random.randn(9, 8, 7, 6)
        input = flow.Tensor(np_arr)
        of_out = input.std(dim=1, unbiased=False, keepdim=False)
        np_out = np.std(np_arr, axis=1)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-04, 1e-04))

    def test_sqrt_tensor_function(test_case):
        input_arr = np.random.rand(1, 6, 3, 8)
        np_out = np.sqrt(input_arr)
        x = flow.Tensor(input_arr)
        of_out = x.sqrt()
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
        )

    def test_rsqrt_tensor_function(test_case):
        np_arr = np.random.rand(3, 2, 5, 7)
        np_out = 1 / np.sqrt(np_arr)
        x = flow.Tensor(np_arr)
        of_out = flow.rsqrt(x)
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
        )

    def test_square_tensor_function(test_case):
        np_arr = np.random.randn(2, 7, 7, 3)
        np_out = np.square(np_arr)
        x = flow.Tensor(np_arr)
        of_out = x.square()
        test_case.assertTrue(
            np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05, equal_nan=True)
        )

    # This test will fail with the rtol and atol constraint under pytorch1.10, but success with pytorch 1.13.
    # The constraints should be removed in the future.
    @autotest(n=5, rtol=1e-3, atol=1e-3)
    def test_addmm_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor(ndim=2, dim0=2, dim1=3).to(device)
        mat1 = random_tensor(ndim=2, dim0=2, dim1=4).to(device)
        mat2 = random_tensor(ndim=2, dim0=4, dim1=3).to(device)
        y = input.addmm(
            mat1,
            mat2,
            beta=random().to(float) | nothing(),
            alpha=random().to(float) | nothing(),
        )
        return y

    # This test will fail with the rtol and atol constraint under pytorch1.10, but success with pytorch 1.13.
    # The constraints should be removed in the future.
    @autotest(n=5, rtol=1e-3, atol=1e-2)
    def test_addmm_broadcast_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor(ndim=2, dim0=1, dim1=1).to(device)
        mat1 = random_tensor(ndim=2, dim0=2, dim1=4).to(device)
        mat2 = random_tensor(ndim=2, dim0=4, dim1=3).to(device)
        y = input.addmm(
            mat1,
            mat2,
            beta=random().to(float) | nothing(),
            alpha=random().to(float) | nothing(),
        )
        return y

    @autotest(n=5)
    def test_clamp_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor(low=-2, high=2).to(device)
        y = input.clamp(
            min=random(low=-1, high=-0.5).to(float),
            max=random(low=0.5, high=1).to(float),
        )
        return y

    @autotest(n=5)
    def test_clamp_inplace_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clamp_(
            min=random(low=-1, high=-0.5).to(float),
            max=random(low=0.5, high=1).to(float),
        )
        return y

    @autotest(auto_backward=False)
    def test_clamp_inplace_tensor_no_grad_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clamp_(
            min=random(low=-1, high=-0.5).to(float),
            max=random(low=0.5, high=1).to(float),
        )
        return y

    @autotest(n=5)
    def test_clamp_minnone_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor(low=-2, high=2).to(device)
        y = input.clamp(
            min=random(low=-1, high=-0.5).to(float) | nothing(),
            max=random(low=0.5, high=1).to(float),
        )
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(auto_backward=False)
    def test_clamp_minnone_tensor_no_grad_with_random_data(test_case):
        device = random_device()
        input = random_tensor(low=-2, high=2).to(device)
        y = input.clamp(
            min=random(low=-1, high=-0.5).to(float) | nothing(),
            max=random(low=0.5, high=1).to(float),
        )
        return y

    @autotest(n=5)
    def test_clamp_inplace_minnone_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clamp_(
            min=random(low=-1, high=-0.5).to(float) | nothing(),
            max=random(low=0.5, high=1).to(float),
        )
        return y

    @autotest(auto_backward=False)
    def test_clamp_inplace_minnone_tensor_no_grad_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clamp_(
            min=random(low=-1, high=-0.5).to(float) | nothing(),
            max=random(low=0.5, high=1).to(float),
        )
        return y

    @autotest(n=5)
    def test_clamp_maxnone_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor(low=-2, high=2).to(device)
        y = input.clamp(
            min=random(low=-1, high=-0.5).to(float),
            max=random(low=0.5, high=1).to(float) | nothing(),
        )
        return y

    @autotest(auto_backward=False)
    def test_clamp_maxnone_tensor_no_grad_with_random_data(test_case):
        device = random_device()
        input = random_tensor(low=-2, high=2).to(device)
        y = input.clamp(
            min=random(low=-1, high=-0.5).to(float),
            max=random(low=0.5, high=1).to(float) | nothing(),
        )
        return y

    @autotest(n=5)
    def test_clamp_inplace_maxnone_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clamp_(
            min=random(low=-1, high=-0.5).to(float),
            max=random(low=0.5, high=1).to(float) | nothing(),
        )
        return y

    @autotest(auto_backward=False)
    def test_clamp_inplace_maxnone_tensor_no_grad_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clamp_(
            min=random(low=-1, high=-0.5).to(float),
            max=random(low=0.5, high=1).to(float) | nothing(),
        )
        return y

    @autotest(n=5)
    def test_clamp_min_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor(low=-2, high=2).to(device)
        y = input.clamp_min(random(low=-0.5, high=0.5).to(float))
        return y

    @autotest(n=5)
    def test_clamp_min_inplace_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clamp_min_(random(low=-0.5, high=0.5).to(float))
        return y

    @autotest(auto_backward=False)
    def test_clamp_min_tensor_no_grad_with_random_data(test_case):
        device = random_device()
        input = random_tensor(low=-2, high=2).to(device)
        y = input.clamp_min(random(low=-0.5, high=0.5).to(float))
        return y

    @autotest(auto_backward=False)
    def test_clamp_min_inplace_tensor_no_grad_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clamp_min_(random(low=-0.5, high=0.5).to(float))
        return y

    @autotest(n=5)
    def test_clamp_max_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor(low=-2, high=2).to(device)
        y = input.clamp_max(random(low=-0.5, high=0.5).to(float))
        return y

    @autotest(n=5)
    def test_clamp_max_inplace_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clamp_max_(random(low=-0.5, high=0.5).to(float))
        return y

    @autotest(auto_backward=False)
    def test_clamp_max_tensor_no_grad_with_random_data(test_case):
        device = random_device()
        input = random_tensor(low=-2, high=2).to(device)
        y = input.clamp_max(random(low=-0.5, high=0.5).to(float))
        return y

    @autotest(auto_backward=False)
    def test_clamp_max_inplace_tensor_no_grad_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clamp_max_(random(low=-0.5, high=0.5).to(float))
        return y

    @autotest(n=5)
    def test_clip_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor(low=-2, high=2).to(device)
        y = input.clip(
            min=random(low=-1, high=-0.5).to(float),
            max=random(low=0.5, high=1).to(float),
        )
        return y

    @autotest(n=5)
    def test_clip_inplace_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clip_(
            min=random(low=-1, high=-0.5).to(float),
            max=random(low=0.5, high=1).to(float),
        )
        return y

    @autotest(n=5)
    def test_clip_minnone_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor(low=-2, high=2).to(device)
        y = input.clip(
            min=random(low=-1, high=-0.5).to(float) | nothing(),
            max=random(low=0.5, high=1).to(float),
        )
        return y

    @autotest(n=5)
    def test_clip_inplace_maxnone_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clip_(
            min=random(low=-1, high=-0.5).to(float),
            max=random(low=0.5, high=1).to(float) | nothing(),
        )
        return y

    @autotest(n=5)
    def test_clip_maxnone_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = input.clip(
            min=random(low=-1, high=-0.5).to(float),
            max=random(low=0.5, high=1).to(float) | nothing(),
        )
        return y

    @autotest(n=5)
    def test_clip_inplace_maxnone_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-2, high=2).to(device)
        y = x + 1
        y.clip_(
            min=random(low=-1, high=-0.5).to(float),
            max=random(low=0.5, high=1).to(float) | nothing(),
        )
        return y

    @autotest(n=5)
    def test_ceil_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = len(input)
        return y

    @autotest(n=5)
    def test_ceil_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = input.ceil()
        return y

    @autotest(n=5)
    def test_expm1_tensor_with_random_data(test_case):
        device = random_device()
        input = random_tensor().to(device)
        y = input.expm1()
        return y

    @autotest(n=5)
    def test_floor_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.floor()
        return y

    @autotest(n=5)
    def test_tensor_var_all_dim_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.var()
        return y

    # TODO(): 'var backward' is composed of several other ops,
    # reducemean doesn't support 0-shape for now
    @autotest(n=5, auto_backward=False)
    def test_tensor_var_one_dim_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = x.var(
            dim=random(low=0, high=4).to(int),
            unbiased=random().to(bool),
            keepdim=random().to(bool),
        )
        return y

    def test_norm_tensor_function(test_case):
        input = flow.tensor(
            np.array([[-4.0, -3.0, -2.0], [-1.0, 0.0, 1.0], [2.0, 3.0, 4.0]]),
            dtype=flow.float32,
        )
        of_out_1 = input.norm("fro")
        np_out_1 = np.linalg.norm(input.numpy(), "fro")
        of_out_2 = input.norm(2, dim=1)
        np_out_2 = np.linalg.norm(input.numpy(), ord=2, axis=1)
        of_out_3 = input.norm(float("inf"), dim=0, keepdim=True)
        np_out_3 = np.linalg.norm(
            input.numpy(), ord=float("inf"), axis=0, keepdims=True
        )
        test_case.assertTrue(np.allclose(of_out_1.numpy(), np_out_1, 1e-05, 1e-05))
        test_case.assertTrue(np.allclose(of_out_2.numpy(), np_out_2, 1e-05, 1e-05))
        test_case.assertTrue(np.allclose(of_out_3.numpy(), np_out_3, 1e-05, 1e-05))

    @autotest(n=5)
    def test_pow_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = random().to(float)
        z = x.pow(y)
        return z

    @autotest(n=5)
    def test_atanh_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-0.5, high=0.49).to(device)
        y = x.atanh()
        return y

    @autotest(n=5)
    def test_acos_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-0.5, high=0.49).to(device)
        y = x.acos()
        return y

    @autotest(n=5)
    def test_acosh_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=2.0, high=3.0).to(device)
        y = x.acosh()
        return y

    @autotest(n=5)
    def test_atan_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.atan()
        return y

    @autotest(n=5)
    def test_arctan_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.arctan()
        return y

    @autotest(n=5)
    def test_tan_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        y = x.tan()
        return y

    @autotest(n=5)
    def test_tan2_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim1=3).to(device)
        y = random_tensor(ndim=2, dim1=3).to(device)
        z = x.atan2(y)
        return z

    @autotest(n=5)
    def test_arctanh_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-0.5, high=0.5).to(device)
        y = x.arctanh()
        return y

    # Not check graph because of one reason:
    # Reason 1, lazy tensor cannot call .numpy(). tensor.numpy() is not allowed to called in nn.Graph.build(*args) or called by lazy tensor.
    # Please refer to File "python/oneflow/nn/modules/nonzero.py", line 29, in nonzero_op.
    @autotest(n=5, auto_backward=False, check_graph="ValidatedFalse")
    def test_tensor_nonzero_with_random_data(test_case):
        device = random_device()
        ndim = random(2, 6).to(int)
        x = random_tensor(ndim=ndim).to(device)
        y = x.nonzero()
        return y

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_tensor_fmod(test_case):
        x = flow.Tensor(np.random.uniform(-100, 100, (5, 5)))
        x.requires_grad = True
        y = np.random.uniform(-10, 10)
        of_out = x.fmod(y)
        np_out = np.sign(x.numpy()) * np.abs(np.fmod(x.numpy(), y))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(
            np.allclose(x.grad.numpy(), np.ones((5, 5)), 0.0001, 0.0001)
        )

    @unittest.skipIf(
        not flow.unittest.env.eager_execution_enabled(),
        "numpy doesn't work in lazy mode",
    )
    def test_magic_fmod(test_case):
        x = flow.Tensor(np.random.uniform(-100, 100, (5, 5)))
        x.requires_grad = True
        y = np.random.uniform(-10, 10)
        of_out = x % y
        np_out = np.sign(x.numpy()) * np.abs(np.fmod(x.numpy(), y))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 0.0001, 0.0001))
        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(
            np.allclose(x.grad.numpy(), np.ones((5, 5)), 0.0001, 0.0001)
        )

    def test_tensor_mish(test_case):
        def np_mish(x):
            f = 1 + np.exp(x)
            y = x * ((f * f - 1) / (f * f + 1))
            y_grad = (f * f - 1) / (f * f + 1) + x * (4 * f * (f - 1)) / (
                (f * f + 1) * (f * f + 1)
            )
            return [y, y_grad]

        np_input = np.random.randn(2, 4, 5, 6)
        of_input = flow.tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_out = of_input.mish()
        (np_out, np_grad) = np_mish(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-05, 1e-05))

    def test_tensor_triu(test_case):
        def np_triu(x, diagonal):
            y = np.triu(x, diagonal)
            y_grad = np.triu(np.ones_like(x), diagonal)
            return [y, y_grad]

        diagonal_list = [2, -1]
        for diagonal in diagonal_list:
            np_input = np.random.randn(2, 4, 6)
            of_input = flow.tensor(np_input, dtype=flow.float32, requires_grad=True)
            of_out = of_input.triu(diagonal)
            (np_out, np_grad) = np_triu(np_input, diagonal)
            test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-05, 1e-05))
            of_out = of_out.sum()
            of_out.backward()
            test_case.assertTrue(
                np.allclose(of_input.grad.numpy(), np_grad, 1e-05, 1e-05)
            )

    def test_tensor_grad_assignment(test_case):
        np_input = np.random.randn(2, 4, 5, 6)
        of_input = flow.tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_output = 2 * of_input
        of_output = of_output.sum()
        of_output.backward()
        new_grad = flow.tensor(
            np.full(np_input.shape, np.random.randn(1)), dtype=flow.float32
        )
        of_input.grad = new_grad
        test_case.assertTrue(
            np.allclose(of_input.grad.detach().numpy(), new_grad.numpy(), 1e-05, 1e-05)
        )
        of_input.grad = None
        test_case.assertTrue(of_input.grad is None)

    def test_tensor_grad_assignment_sum(test_case):
        np_input = np.random.randn(1, 5, 7, 3)
        of_input = flow.tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_output = of_input.sum()
        of_output.backward()
        rand_init = np.random.randn(1)
        rand_scale = np.random.randn(1)
        new_grad = flow.tensor(np.full(np_input.shape, rand_init), dtype=flow.float32)
        of_input.grad = new_grad
        of_output = flow.tensor(rand_scale, dtype=flow.float32) * of_input
        of_output = of_output.sum()
        of_output.backward()
        test_case.assertTrue(
            np.allclose(
                of_input.grad.detach().numpy(),
                np.full(np_input.shape, rand_init + rand_scale),
                1e-05,
                1e-05,
            )
        )
        of_input.grad = of_input.grad * 2
        test_case.assertTrue(
            np.allclose(
                of_input.grad.detach().numpy(),
                2 * np.full(np_input.shape, rand_init + rand_scale),
                1e-05,
                1e-05,
            )
        )

    def test_tensor_mish(test_case):
        def np_mish(x):
            f = 1 + np.exp(x)
            y = x * ((f * f - 1) / (f * f + 1))
            y_grad = (f * f - 1) / (f * f + 1) + x * (4 * f * (f - 1)) / (
                (f * f + 1) * (f * f + 1)
            )
            return [y, y_grad]

        np_input = np.random.randn(2, 4, 5, 6,)
        of_input = flow.tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_out = of_input.mish()

        np_out, np_grad = np_mish(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-5, 1e-5))

    def test_tensor_silu(test_case):
        def np_silu(x):
            _sig = 1 / (1 + np.exp(-x))
            y = x * _sig
            y_grad = _sig * (1 + x * (1 - _sig))
            return [y, y_grad]

        np_input = np.random.randn(2, 4, 5, 6,)
        of_input = flow.tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_out = of_input.silu()

        np_out, np_grad = np_silu(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-5, 1e-5))

    def test_tensor_selu(test_case):
        _scale = 1.0507009873554804934193349852946
        _alpha = 1.6732632423543772848170429916717

        def np_selu(x):
            y = np.where(x < 0, _scale * _alpha * (np.exp(x) - 1), _scale * x)
            y_grad = np.where(x < 0, _scale * _alpha * np.exp(x), _scale)
            return [y, y_grad]

        np_input = np.random.randn(2, 4, 5, 6,)
        of_input = flow.tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_out = of_input.selu()

        np_out, np_grad = np_selu(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-5, 1e-5))

    @unittest.skip("still have error in ci")
    def test_tensor_softsign(test_case):
        def np_softsign(x):
            y = x / (1 + np.abs(x))
            y_grad = 1 / np.square(1 + np.abs(x))
            return [y, y_grad]

        np_input = np.random.randn(2, 4, 5, 6,)
        of_input = flow.tensor(np_input, dtype=flow.float32, requires_grad=True)
        of_out = of_input.softsign()

        np_out, np_grad = np_softsign(np_input)
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out, 1e-5, 1e-5))

        of_out = of_out.sum()
        of_out.backward()
        test_case.assertTrue(np.allclose(of_input.grad.numpy(), np_grad, 1e-5, 1e-5))

    @autotest(auto_backward=False)
    def test_eq_tensor_with_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        y = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        return x.eq(y)

    @autotest(auto_backward=False)
    def test_eq_tensor_with_same_random_data(test_case):
        device = random_device()
        shape = random_tensor().oneflow.shape
        x = random_tensor(len(shape), *shape, requires_grad=False).to(device)
        return x.eq(x)

    @autotest(n=5)
    def test_erf_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return x.erf()

    @autotest(n=5)
    def test_erfc_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return x.erfc()

    @autotest(
        auto_backward=False
    )  # Todo: After add gradient func, you should set `auto_backward` as True
    def test_erfinv_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-1, high=1).to(device).requires_grad_(False)
        return x.erfinv()

    @autotest(
        n=10, auto_backward=False
    )  # Todo: After add gradient func, you should set `auto_backward` as True
    def test_erfinv_inplace_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor(low=-1, high=1).to(device).requires_grad_(False)
        y = x + 1
        y.erfinv_()
        return y

    @autotest(n=5)
    def test_exp_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return x.exp()

    @autotest(n=5)
    def test_exp2_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return x.exp2()

    @autotest(n=5)
    def test_round_tensor_with_random_data(test_case):
        device = random_device()
        x = random_tensor().to(device)
        return x.round()

    @autotest(n=5)
    def test_tensor_diag_one_dim(test_case):
        device = random_device()
        x = random_tensor(ndim=1, dim0=random()).to(device)
        return x.diag()

    @autotest(n=5)
    def test_flow_tensor_expand_with_random_data(test_case):
        random_expand_size = random(1, 6).to(int).value()
        x = random_tensor(ndim=5, dim0=1, dim1=1, dim2=1, dim3=1, dim4=1)
        ndim = 5
        expand_size = random_expand_size
        dim_size = [1,] * ndim
        random_index = random(0, ndim).to(int).value()
        dim_size[random_index] = expand_size
        return x.expand(*dim_size)

    @autotest(n=5)
    def test_flow_tensor_expand_with_random_data(test_case):
        random_expand_size = random(1, 6).to(int).value()
        x = random_tensor(ndim=5, dim0=1, dim1=1, dim2=1, dim3=1, dim4=1)
        ndim = 5
        expand_size = random_expand_size
        dim_size = [1,] * ndim
        random_index = random(0, ndim).to(int).value()
        dim_size[random_index] = expand_size
        y = torch.ones(dim_size)
        return x.expand_as(y)

    @autotest(n=5)
    def test_flow_tensor_view_with_random_data(test_case):
        dim0_ = random(2, 4).to(int)
        dim1_ = random(2, 4).to(int)
        dim2_ = random(2, 4).to(int)
        dim3_ = random(2, 4).to(int)
        dim4_ = random(2, 4).to(int)
        x = random_tensor(
            ndim=5, dim0=dim0_, dim1=dim1_, dim2=dim2_, dim3=dim3_, dim4=dim4_
        )
        shape = [x.value() for x in [dim4_, dim3_, dim2_, dim1_, dim0_]]
        return [x.view(shape), x.view(size=shape)]

    @autotest(n=5)
    def test_flow_tensor_view_as_with_random_data(test_case):
        dim0_ = random(2, 4).to(int)
        dim1_ = random(2, 4).to(int)
        dim2_ = random(2, 4).to(int)
        dim3_ = random(2, 4).to(int)
        dim4_ = random(2, 4).to(int)
        x = random_tensor(
            ndim=5, dim0=dim0_, dim1=dim1_, dim2=dim2_, dim3=dim3_, dim4=dim4_
        )
        other = random_tensor(
            ndim=5, dim0=dim4_, dim1=dim3_, dim2=dim2_, dim3=dim1_, dim4=dim0_
        )
        return x.view_as(other)

    @autotest(n=5)
    def test_tensor_diag_other_dim(test_case):
        device = random_device()
        x = random_tensor(ndim=2, dim0=random(), dim1=random()).to(device)
        return x.diag()

    @autotest(auto_backward=False)
    def test_floordiv_elementwise_tensor_with_random_data(test_case):
        device = random_device()
        # The random value is narrowed to positive number because of the error from pytorch 1.10.0
        # Please remove the value range striction after updating the pytorch version of ci to 1.13.
        input = random_tensor(ndim=2, dim0=4, dim1=8, low=0, high=10).to(device)
        other = random_tensor(ndim=2, dim0=4, dim1=8, low=0, high=10).to(device)
        y = input.floor_divide(other)
        return y

    @autotest(auto_backward=False)
    def test_scalar_floordiv_tensor_with_random_data(test_case):
        device = random_device()
        # The random value is narrowed to positive number because of the error from pytorch 1.10.0
        # Please remove the value range striction after updating the pytorch version of ci to 1.13.
        input = random_tensor(ndim=2, dim0=4, dim1=8, low=0, high=10).to(device)
        other = random().to(int)
        y = input.floor_divide(other)
        return y

    @flow.unittest.skip_unless_1n4d()
    def test_construct_global_tensor_by_numpy(test_case):
        x = np.ones((4, 4), dtype=np.int32)
        placement = flow.placement("cuda", [0, 1, 2, 3])
        y = flow.tensor(
            x,
            dtype=flow.float32,
            placement=placement,
            sbp=[flow.sbp.split(0)],
            requires_grad=False,
        )
        test_case.assertTrue(y.dtype == flow.float32)
        test_case.assertTrue(
            np.allclose(y.to_local().numpy(), np.ones((1, 4), dtype=np.float32))
        )
        test_case.assertEqual(y.placement, placement)

        y_default_dtype = flow.tensor(
            x, placement=placement, sbp=[flow.sbp.split(0)], requires_grad=False,
        )
        test_case.assertTrue(y_default_dtype.dtype == flow.int32)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestTensorNumpy(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_1d_sbp_tensor_numpy_1n2d(test_case):
        ori_x = flow.tensor([1, 2, 3, 4]) + flow.env.get_rank()
        placement = flow.placement.all("cpu")
        x = ori_x.to_global(placement=placement, sbp=flow.sbp.split(0))
        test_case.assertTrue(np.allclose(x.numpy(), [1, 2, 3, 4, 2, 3, 4, 5]))

        x = ori_x.to_global(placement=placement, sbp=flow.sbp.broadcast, copy=True)
        test_case.assertTrue(np.allclose(x.numpy(), [1, 2, 3, 4]))

        x = ori_x.to_global(placement=placement, sbp=flow.sbp.partial_sum)
        test_case.assertTrue(np.allclose(x.numpy(), [3, 5, 7, 9]))

        placement = flow.placement.all("cuda")
        x = ori_x.to_global(placement=placement, sbp=flow.sbp.split(0))
        test_case.assertTrue(np.allclose(x.numpy(), [1, 2, 3, 4, 2, 3, 4, 5]))

        x = ori_x.to_global(placement=placement, sbp=flow.sbp.broadcast, copy=True)
        test_case.assertTrue(np.allclose(x.numpy(), [1, 2, 3, 4]))

        x = ori_x.to_global(placement=placement, sbp=flow.sbp.partial_sum)
        test_case.assertTrue(np.allclose(x.numpy(), [3, 5, 7, 9]))

    @flow.unittest.skip_unless_1n2d()
    def test_2d_sbp_tensor_numpy_1n2d(test_case):
        ori_x = flow.tensor(np.ones((2, 2))) + flow.env.get_rank()
        placement = flow.placement("cuda", [[0], [1]])
        x = ori_x.to_global(
            placement=placement, sbp=[flow.sbp.split(0), flow.sbp.split(1)]
        )
        test_case.assertTrue(np.allclose(x.numpy(), [[1, 1], [1, 1], [2, 2], [2, 2]]))

        x = ori_x.to_global(
            placement=placement, sbp=[flow.sbp.broadcast, flow.sbp.split(0)]
        )
        test_case.assertTrue(np.allclose(x.numpy(), [[1, 1], [1, 1]]))

        x = ori_x.to_global(
            placement=placement,
            sbp=[flow.sbp.partial_sum, flow.sbp.broadcast],
            copy=True,
        )
        test_case.assertTrue(np.allclose(x.numpy(), [[3, 3], [3, 3]]))

    @flow.unittest.skip_unless_1n4d()
    def test_2d_sbp_tensor_numpy_1n4d(test_case):
        ori_x = flow.tensor(np.ones((2, 2))) + flow.env.get_rank()
        placement = flow.placement("cuda", [[0, 1], [2, 3]])

        x = ori_x.to_global(
            placement=placement, sbp=[flow.sbp.split(0), flow.sbp.split(1)]
        )
        test_case.assertTrue(
            np.allclose(
                x.numpy(), [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
            )
        )

        x = ori_x.to_global(
            placement=placement, sbp=[flow.sbp.split(0), flow.sbp.partial_sum]
        )
        test_case.assertTrue(np.allclose(x.numpy(), [[3, 3], [3, 3], [7, 7], [7, 7]]))

        # TODO: (s0, b) has bug
        # x = ori_x.to_global(placement=placement, sbp=[flow.sbp.split(0), flow.sbp.broadcast])

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_tensor_bmm(test_case):
        t = random(1, 5)
        k = random(1, 5)
        input1 = random_tensor(ndim=3, dim0=t, dim1=3, dim2=k)
        input2 = random_tensor(ndim=3, dim0=t, dim1=k, dim2=5)
        of_out = input1.bmm(input2)
        return of_out

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_tensor_split(test_case):
        k0 = random(2, 6)
        k1 = random(2, 6)
        k2 = random(2, 6)
        rand_dim = random(0, 3).to(int)
        device = random_device()
        x = random_tensor(ndim=3, dim0=k0, dim1=k1, dim2=k2).to(device)
        res = x.split(2, dim=rand_dim)
        return torch.cat(res, rand_dim)

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_tensor_split_sizes(test_case):
        k0 = random(2, 6)
        k1 = 7
        k2 = random(2, 6)
        device = random_device()
        x = random_tensor(ndim=3, dim0=k0, dim1=k1, dim2=k2).to(device)
        res = x.split([1, 2, 3, 1], dim=-2)
        return torch.cat(res, dim=1)

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_tensor_unbind(test_case):
        device = random_device()
        x = random_tensor(ndim=4).to(device)
        y = x.unbind(random(0, 4).to(int))
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_tensor_swapaxes(test_case):
        device = random_device()
        x = random_tensor(ndim=3).to(device)
        y = x.swapaxes(random(0, 2).to(int), random(0, 2).to(int))
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_tensor_swapdimst(test_case):
        device = random_device()
        x = random_tensor(ndim=3).to(device)
        y = x.swapdims(random(0, 3).to(int), random(0, 3).to(int))
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_tensor_int_repeat_interleave_dim_none(test_case):
        x = random_tensor(ndim=2, dim0=1, dim1=2)
        y = x.repeat_interleave(2)
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_tensor_int_repeat_interleave_with_dim(test_case):
        x = random_tensor(ndim=3, dim0=2, dim1=2, dim2=3)
        dim = random(low=0, high=2).to(int)
        y = x.repeat_interleave(2, dim)
        return y

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_tensor_tensor_repeat_interleave_dim(test_case):
        x = random_tensor(ndim=3, dim0=2, dim1=2, dim2=3)
        y = random_tensor(ndim=1, dim0=2, dtype=int, low=1, high=4)
        z = x.repeat_interleave(y, 1)
        return z

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5, rtol=1e-3)
    def test_tensor_tensor_repeat_interleave_dim_with_output_size(test_case):
        x = random_tensor(ndim=3, dim0=2, dim1=2, dim2=3)
        y = random_tensor(ndim=1, dim0=2, dtype=int, low=1, high=4)
        z = x.repeat_interleave(y, 1, output_size=2)
        return z

    @flow.unittest.skip_unless_1n2d()
    @globaltest
    def test_global_tensor_detach(test_case):
        device = random_device().value()
        placement = flow.placement(device, [0, 1])
        a = flow.ones(4, 8).to_global(placement, flow.sbp.broadcast)
        test_case.assertTrue(a.is_leaf)
        b = a.float().clone().detach()
        test_case.assertTrue(b.is_leaf)

    @flow.unittest.skip_unless_1n1d()
    @autotest(n=5)
    def test_tensor_nansum(test_case):
        device = random_device()
        x = random_tensor(4, random(0, 5), 2).to(device)
        mask = x < 0
        x = x.masked_fill(mask, float("nan"))
        y = x.nansum()
        return y


if __name__ == "__main__":
    unittest.main()
