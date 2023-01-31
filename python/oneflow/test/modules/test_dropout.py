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
from collections import OrderedDict

import numpy as np
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *


def do_test_dropout_numpy_p0(test_case, shape, device, dtype):
    np_x = np.random.randn(*shape).astype(dtype)
    np_one_mask = np.ones_like(np_x)
    x_tensor = flow.tensor(np_x, requires_grad=True, device=device)
    out = flow._C.dropout(x_tensor, p=0.0)
    test_case.assertTrue(np.allclose(out.numpy(), np_x, atol=1e-5, rtol=1e-5))
    out_sum = out.sum()
    out_sum.backward()
    test_case.assertTrue(
        np.allclose(x_tensor.grad.numpy(), np_one_mask, atol=1e-5, rtol=1e-5)
    )


def do_test_dropout_numpy_p1(test_case, shape, device, dtype):
    np_x = np.random.randn(*shape).astype(dtype)
    np_zero_mask = np.zeros_like(np_x)
    x_tensor = flow.tensor(np_x, requires_grad=True, device=device)
    out = flow._C.dropout(x_tensor, p=1.0)
    test_case.assertTrue(np.allclose(out.numpy(), np_zero_mask, atol=1e-5, rtol=1e-5))
    out_sum = out.sum()
    out_sum.backward()
    test_case.assertTrue(
        np.allclose(x_tensor.grad.numpy(), np_zero_mask, atol=1e-5, rtol=1e-5)
    )


def do_test_dropout_numpy_fp16_p0(test_case, shape):
    np_x = np.random.randn(*shape).astype(np.float32)
    np_x_fp16 = np_x.astype(np.float16)
    x_tensor = flow.tensor(np_x, requires_grad=True, device="cuda")
    x_tensor_fp16 = flow.cast(x_tensor, flow.float16)
    np_one_mask = np.ones_like(np_x)
    out = flow._C.dropout(x_tensor_fp16, p=0.0)
    out_fp32 = flow.cast(out, flow.float32)
    test_case.assertTrue(np.allclose(out_fp32.numpy(), np_x_fp16, atol=1e-5, rtol=1e-5))
    out_sum = out_fp32.sum()
    out_sum.backward()
    test_case.assertTrue(
        np.allclose(x_tensor.grad.numpy(), np_one_mask, atol=1e-5, rtol=1e-5)
    )


def do_test_dropout_numpy_fp16_p1(test_case, shape):
    np_x = np.random.randn(*shape).astype(np.float32)
    x_tensor = flow.tensor(np_x, requires_grad=True, device="cuda")
    x_tensor_fp16 = flow.cast(x_tensor, flow.float16)
    np_zero_mask = np.zeros_like(np_x)
    out = flow._C.dropout(x_tensor_fp16, p=1.0)
    out_fp32 = flow.cast(out, flow.float32)
    test_case.assertTrue(
        np.allclose(out_fp32.numpy(), np_zero_mask, atol=1e-5, rtol=1e-5)
    )
    out_sum = out_fp32.sum()
    out_sum.backward()
    test_case.assertTrue(
        np.allclose(x_tensor.grad.numpy(), np_zero_mask, atol=1e-5, rtol=1e-5)
    )


def do_test_dropout_addend_numpy_p0(test_case, shape, device, dtype):
    np_x = np.random.randn(*shape).astype(dtype)
    np_addend = np.random.randn(*shape).astype(dtype)
    np_one_mask = np.ones_like(np_x)
    x_tensor = flow.tensor(np_x, requires_grad=True, device=device)
    addend_tensor = flow.tensor(np_addend, requires_grad=True, device=device)
    DropoutModule = flow.nn.Dropout(p=0.0)
    out = DropoutModule(x_tensor, addend_tensor)
    test_case.assertTrue(
        np.allclose(out.numpy(), np_x + np_addend, atol=1e-5, rtol=1e-5)
    )
    out_sum = out.sum()
    out_sum.backward()
    test_case.assertTrue(
        np.allclose(x_tensor.grad.numpy(), np_one_mask, atol=1e-5, rtol=1e-5)
    )
    test_case.assertTrue(
        np.allclose(addend_tensor.grad.numpy(), np_one_mask, atol=1e-5, rtol=1e-5)
    )


def do_test_dropout_addend_numpy_p1(test_case, shape, device, dtype):
    np_x = np.random.randn(*shape).astype(dtype)
    np_addend = np.random.randn(*shape).astype(dtype)
    np_one_mask = np.ones_like(np_x)
    np_zero_mask = np.zeros_like(np_x)
    x_tensor = flow.tensor(np_x, requires_grad=True, device=device)
    addend_tensor = flow.tensor(np_addend, requires_grad=True, device=device)
    DropoutModule = flow.nn.Dropout(p=1.0)
    out = DropoutModule(x_tensor, addend_tensor)
    test_case.assertTrue(np.allclose(out.numpy(), np_addend, atol=1e-5, rtol=1e-5))
    out_sum = out.sum()
    out_sum.backward()
    test_case.assertTrue(
        np.allclose(x_tensor.grad.numpy(), np_zero_mask, atol=1e-5, rtol=1e-5)
    )
    test_case.assertTrue(
        np.allclose(addend_tensor.grad.numpy(), np_one_mask, atol=1e-5, rtol=1e-5)
    )


def do_test_dropout_addend_numpy_fp16_p0(test_case, shape):
    np_x = np.random.randn(*shape).astype(np.float32)
    np_x_fp16 = np_x.astype(np.float16)
    np_addend = np.random.randn(*shape).astype(np.float32)
    np_addend_fp16 = np_addend.astype(np.float16)
    x_tensor = flow.tensor(np_x, requires_grad=True, device="cuda")
    x_tensor_fp16 = flow.cast(x_tensor, flow.float16)
    addend_tensor = flow.tensor(np_addend, requires_grad=True, device="cuda")
    addend_tensor_fp16 = flow.cast(addend_tensor, flow.float16)
    np_one_mask = np.ones_like(np_x)
    DropoutModule = flow.nn.Dropout(p=0.0)
    out = DropoutModule(x_tensor_fp16, addend_tensor_fp16)
    out_fp32 = flow.cast(out, flow.float32)
    test_case.assertTrue(
        np.allclose(out_fp32.numpy(), np_x_fp16 + np_addend_fp16, atol=1e-5, rtol=1e-5)
    )
    out_sum = out_fp32.sum()
    out_sum.backward()
    test_case.assertTrue(
        np.allclose(x_tensor.grad.numpy(), np_one_mask, atol=1e-5, rtol=1e-5)
    )
    test_case.assertTrue(
        np.allclose(addend_tensor.grad.numpy(), np_one_mask, atol=1e-5, rtol=1e-5)
    )


def do_test_dropout_addend_numpy_fp16_p1(test_case, shape):
    np_x = np.random.randn(*shape).astype(np.float32)
    np_addend = np.random.randn(*shape).astype(np.float32)
    np_addend_fp16 = np_addend.astype(np.float16)
    x_tensor = flow.tensor(np_x, requires_grad=True, device="cuda")
    x_tensor_fp16 = flow.cast(x_tensor, flow.float16)
    addend_tensor = flow.tensor(np_addend, requires_grad=True, device="cuda")
    addend_tensor_fp16 = flow.cast(addend_tensor, flow.float16)
    np_zero_mask = np.zeros_like(np_x)
    np_one_mask = np.ones_like(np_x)
    DropoutModule = flow.nn.Dropout(p=1.0)
    out = DropoutModule(x_tensor_fp16, addend_tensor_fp16)
    out_fp32 = flow.cast(out, flow.float32)
    test_case.assertTrue(
        np.allclose(out_fp32.numpy(), np_addend_fp16, atol=1e-5, rtol=1e-5)
    )
    out_sum = out_fp32.sum()
    out_sum.backward()
    test_case.assertTrue(
        np.allclose(x_tensor.grad.numpy(), np_zero_mask, atol=1e-5, rtol=1e-5)
    )
    test_case.assertTrue(
        np.allclose(addend_tensor.grad.numpy(), np_one_mask, atol=1e-5, rtol=1e-5)
    )


def fixed_cpu_seed_dropout_test(test_case):
    gen1 = flow.Generator()
    gen1.manual_seed(5)
    dropped_array1 = np.array(
        [
            [0.000000, 0.000000, 1.333333],
            [1.333333, 0.000000, 1.333333],
            [1.333333, 1.333333, 1.333333],
        ]
    ).astype(np.float32)
    dropout1 = flow.nn.Dropout(p=0.25, generator=gen1)
    x = flow.ones((3, 3), dtype=flow.float32)
    out1 = dropout1(x)
    test_case.assertTrue(
        np.allclose(out1.numpy(), dropped_array1, atol=1e-4, rtol=1e-4)
    )
    gen2 = flow.Generator()
    gen2.manual_seed(7)
    dropout2 = flow.nn.Dropout(p=0.5, generator=gen2)
    dropped_array2 = np.array(
        [[0.0, 0.0, 2.0], [0.0, 0.0, 2.0], [2.0, 0.0, 2.0]]
    ).astype(np.float32)
    out2 = dropout2(x)
    test_case.assertTrue(
        np.allclose(out2.numpy(), dropped_array2, atol=1e-4, rtol=1e-4)
    )


def fixed_gpu_seed_dropout_test(test_case):
    gen1 = flow.Generator()
    gen1.manual_seed(5)
    dropped_array1 = np.array(
        [[1.2500, 0.0000, 1.2500], [1.2500, 1.2500, 1.2500], [1.2500, 1.2500, 1.2500]]
    ).astype(np.float32)
    dropout1 = flow.nn.Dropout(p=0.2, generator=gen1).to("cuda")
    x = flow.ones((3, 3), dtype=flow.float32).to("cuda")
    out1 = dropout1(x)
    test_case.assertTrue(
        np.allclose(out1.numpy(), dropped_array1, atol=1e-4, rtol=1e-4)
    )
    gen2 = flow.Generator()
    gen2.manual_seed(7)
    dropout2 = flow.nn.Dropout(p=0.7, generator=gen2).to("cuda")
    dropped_array2 = np.array(
        [
            [3.333333, 3.333333, 0.000000],
            [0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 0.000000],
        ]
    ).astype(np.float32)
    out2 = dropout2(x)
    test_case.assertTrue(
        np.allclose(out2.numpy(), dropped_array2, atol=1e-4, rtol=1e-4)
    )


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    def test_dropout_numpy_case(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [do_test_dropout_numpy_p0, do_test_dropout_numpy_p1]
        arg_dict["shape"] = [[4], [4, 3], [4, 127, 256], [2, 1024, 1024]]
        arg_dict["device"] = ["cuda"]
        if os.getenv("ONEFLOW_TEST_CPU_ONLY"):
            arg_dict["device"] = ["cpu"]
        arg_dict["dtype"] = [np.float32, np.float64]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dropout_fp16_numpy_case(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            do_test_dropout_numpy_fp16_p0,
            do_test_dropout_numpy_fp16_p1,
        ]
        arg_dict["shape"] = [[4, 127, 256], [5, 63, 49], [7, 32, 64], [16, 512, 512]]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_dropout_addend_numpy_case(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            do_test_dropout_addend_numpy_p0,
            do_test_dropout_addend_numpy_p1,
        ]
        arg_dict["shape"] = [[4, 47, 156], [5, 33, 65], [3, 132, 94], [9, 256, 63]]
        arg_dict["device"] = ["cpu", "cuda"]
        arg_dict["dtype"] = [np.float32, np.float64]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_dropout_addend_fp16_numpy_case(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            do_test_dropout_addend_numpy_fp16_p0,
            do_test_dropout_addend_numpy_fp16_p1,
        ]
        arg_dict["shape"] = [[2, 44, 66], [1, 2, 7], [5, 32, 74], [8, 125, 63]]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_cpu_fixed_dropout(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            fixed_cpu_seed_dropout_test,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case)

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_gpu_fixed_dropout(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            fixed_gpu_seed_dropout_test,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case)

    @autotest(n=5)
    def test_dropout_p0(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        m = torch.nn.Dropout(p=0, inplace=False)
        return m(x)

    @unittest.skipIf(True, "Pytorch 1.10.0 do not have Dropout1d module")
    @autotest(n=5)
    def test_dropout1d_p0(test_case):
        device = random_device()
        x = random_tensor(ndim=random(2, 4), dim0=random(1, 8)).to(device)
        m = torch.nn.Dropout1d(p=0, inplace=False)
        return m(x)

    @autotest(n=5)
    def test_dropout2d_p0(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        m = torch.nn.Dropout2d(p=0, inplace=False)
        return m(x)

    @unittest.skipIf(
        True,
        "this will Pytorch 1.13.0, but failed with Pytorch 1.10.0 because some non-leaf tensors don't have grad",
    )
    @autotest(n=5)
    def test_dropout3d_p0(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        m = torch.nn.Dropout3d(p=0, inplace=False)
        return m(x)

    @autotest(n=5)
    def test_dropout_p1(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        m = torch.nn.Dropout(p=1.0, inplace=False)
        return m(x)

    @unittest.skipIf(True, "Pytorch 1.10.0 do not have Dropout1d module")
    @autotest(n=5)
    def test_dropout1d_p1(test_case):
        device = random_device()
        x = random_tensor(ndim=random(2, 4), dim0=random(1, 8)).to(device)
        m = torch.nn.Dropout1d(p=1.0, inplace=False)
        return m(x)

    @autotest(n=5)
    def test_dropout2d_p1(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        m = torch.nn.Dropout2d(p=1.0, inplace=False)
        return m(x)

    @unittest.skipIf(
        True,
        "this will Pytorch 1.13.0, but failed with Pytorch 1.10.0 because some non-leaf tensors don't have grad",
    )
    @autotest(n=5)
    def test_dropout3d_p1(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        m = torch.nn.Dropout3d(p=1.0, inplace=False)
        return m(x)

    @unittest.skipIf(True, "Pytorch 1.10.0 do not have Dropout1d module")
    @autotest(n=5)
    def test_functional_dropout1d_p1(test_case):
        device = random_device()
        x = random_tensor(ndim=random(2, 4), dim0=random(1, 8)).to(device)
        return torch.nn.functional.dropout1d(x, p=1.0)

    @autotest(n=5)
    def test_functional_dropout2d_p1(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        return torch.nn.functional.dropout2d(x, p=1.0)

    @unittest.skipIf(
        True,
        "this will Pytorch 1.13.0, but failed with Pytorch 1.10.0 because some non-leaf tensors don't have grad",
    )
    @autotest(n=5)
    def test_functional_dropout3d_p1(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        return torch.nn.functional.dropout3d(x, p=1.0)

    @autotest(n=5, check_graph=False)
    def test_dropout_eval(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        m = torch.nn.Dropout(p=1.0, inplace=False)
        m.eval()
        return m(x)

    @unittest.skipIf(True, "Pytorch 1.10.0 do not have Dropout1d module")
    @autotest(n=5, check_graph=False)
    def test_dropout1d_eval(test_case):
        device = random_device()
        x = random_tensor(ndim=random(2, 4), dim0=random(1, 8)).to(device)
        m = torch.nn.Dropout1d(p=1.0, inplace=False)
        m.eval()
        return m(x)

    @autotest(n=5, check_graph=False)
    def test_dropout2d_eval(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        m = torch.nn.Dropout2d(p=1.0, inplace=False)
        m.eval()
        return m(x)

    @unittest.skipIf(
        True,
        "this will Pytorch 1.13.0, but failed with Pytorch 1.10.0 because some non-leaf tensors don't have grad",
    )
    @autotest(n=5, check_graph=False)
    def test_dropout3d_eval(test_case):
        device = random_device()
        x = random_tensor(ndim=random(), dim0=random(1, 8)).to(device)
        m = torch.nn.Dropout3d(p=1.0, inplace=False)
        m.eval()
        return m(x)

    @autotest(n=5, check_graph=False)
    def test_0dim_dropout_eval(test_case):
        device = random_device()
        x = random_tensor(ndim=0).to(device)
        m = torch.nn.Dropout(p=1.0, inplace=False)
        m.eval()
        return m(x)

    @profile(torch.nn.functional.dropout)
    def profile_dropout(test_case):
        input = torch.ones(100, 128)
        torch.nn.functional.dropout(input, p=0.3)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n2d()
class TestDropoutOnNonDefaultDevice(flow.unittest.TestCase):
    def test_non_default_device(test_case):
        x = flow.tensor([2, 3], dtype=flow.float, device="cuda:1")
        y = flow._C.dropout(x)
        test_case.assertEqual(y.device, flow.device("cuda:1"))


if __name__ == "__main__":
    unittest.main()
