"""
Copyright 2023 The OneFlow Authors. All rights reserved.
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
import torch

# import oneflow.unittest
# from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

import oneflow as flow


def tensor_builder(params: dict, dtype=np.complex64):
    input_shape = params["shape"]

    # generate random input
    if dtype in [np.complex64, np.complex128]:
        x = np.random.randn(*input_shape) + 1.0j * np.random.randn(*input_shape)
        x = x.astype(dtype)
    else:
        x = np.random.randn(*input_shape).astype(dtype)

    # requires grad
    x_flow = flow.from_numpy(x).requires_grad_(True)
    x_torch = torch.from_numpy(x).requires_grad_(True)
    # x_flow = flow.from_numpy(x).requires_grad_(False)
    # x_torch = torch.from_numpy(x).requires_grad_(False)

    return x_flow, x_torch


def compare_result(test_case, a, b, rtol=1e-5, atol=1e-8):
    test_case.assertTrue(
        np.allclose(a.numpy(), b.numpy(), rtol=rtol, atol=atol),
        f"\na\n{a.numpy()}\n{'-' * 80}\nb:\n{b.numpy()}\n{'*' * 80}\ndiff:\n{a.numpy() - b.numpy()}",
    )


def _test_fft(test_case, dtype=np.complex64, params: dict = None):
    print(f"========== Start Testing ==========")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")

    x_flow, x_torch = tensor_builder(params=params, dtype=dtype)
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    print(f"fft n: {n}")
    print(f"fft dim: {dim}")
    print(f"fft norm: {norm}")
    print(f"x_flow.dtype: {x_flow.dtype}")
    print("x_torch.dtype: ", x_torch.dtype)

    # forward
    y_torch = torch.fft.fft(x_torch, n=n, dim=dim, norm=norm)
    y_torch_sum = y_torch.sum()

    # backward
    y_torch_sum.backward()

    # copy back to cpu memory
    x_torch_grad = x_torch.grad.detach().cpu()
    y_torch = y_torch.detach().cpu()

    # forward
    y_flow = flow._C.fft(x_flow, n=n, dim=dim, norm=norm)
    y_flow_sum = y_flow.sum()

    # backward
    y_flow_sum.backward()

    # copy back to cpu memory
    x_flow_grad = x_flow.grad.detach().cpu()
    y_flow = y_flow.detach().cpu()
    if torch.is_conj(y_torch):
        y_torch = torch.resolve_conj(y_torch)
    if torch.is_conj(x_torch_grad):
        x_torch_grad = torch.resolve_conj(x_torch_grad)

    compare_result(test_case, y_flow, y_torch, 1e-5, 1e-2)
    compare_result(test_case, x_flow_grad, x_torch_grad, 1e-5, 1e-2)

    print(f"============== PASSED =============")
    print("\n")


def _test_ifft(test_case, dtype=np.complex64, params: dict = None):
    print(f"========== Start Testing ==========")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")

    x_flow, x_torch = tensor_builder(params=params, dtype=dtype)
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    print(f"fft n: {n}")
    print(f"fft dim: {dim}")
    print(f"fft norm: {norm}")
    print(f"x_flow.dtype: {x_flow.dtype}")
    print("x_torch.dtype: ", x_torch.dtype)

    # forward
    y_torch = torch.fft.ifft(x_torch, n=n, dim=dim, norm=norm)
    y_torch_sum = y_torch.sum()

    # backward
    y_torch_sum.backward()

    # copy back to cpu memory
    x_torch_grad = x_torch.grad.detach().cpu()
    y_torch = y_torch.detach().cpu()

    # forward
    y_flow = flow._C.ifft(x_flow, n=n, dim=dim, norm=norm)
    y_flow_sum = y_flow.sum()

    # backward
    y_flow_sum.backward()

    # copy back to cpu memory
    x_flow_grad = x_flow.grad.detach().cpu()
    y_flow = y_flow.detach().cpu()
    if torch.is_conj(y_torch):
        y_torch = torch.resolve_conj(y_torch)
    if torch.is_conj(x_torch_grad):
        x_torch_grad = torch.resolve_conj(x_torch_grad)

    compare_result(test_case, y_flow, y_torch, 1e-5, 1e-2)
    compare_result(test_case, x_flow_grad, x_torch_grad, 1e-5, 1e-2)

    print(f"============== PASSED =============")
    print("\n")


def _test_rfft(test_case, dtype=np.float32, params: dict = None):
    print(f"========== Start Testing ==========")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")

    x_flow, x_torch = tensor_builder(params=params, dtype=dtype)
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    print(f"rfft n: {n}")
    print(f"rfft dim: {dim}")
    print(f"rfft norm: {norm}")
    print(f"x_flow.dtype: {x_flow.dtype}")
    print("x_torch.dtype: ", x_torch.dtype)

    # forward
    y_torch = torch.fft.rfft(x_torch, n=n, dim=dim, norm=norm)
    y_torch_sum = y_torch.sum()

    # backward
    y_torch_sum.backward()

    # copy back to cpu memory
    x_torch_grad = x_torch.grad.detach().cpu()
    y_torch = y_torch.detach().cpu()

    # forward
    y_flow = flow._C.rfft(x_flow, n=n, dim=dim, norm=norm)

    y_flow_sum = y_flow.sum()

    # backward
    y_flow_sum.backward()

    # copy back to cpu memory
    x_flow_grad = x_flow.grad.detach().cpu()
    y_flow = y_flow.detach().cpu()
    if torch.is_conj(y_torch):
        y_torch = torch.resolve_conj(y_torch)
    if torch.is_conj(x_torch_grad):
        x_torch_grad = torch.resolve_conj(x_torch_grad)

    compare_result(test_case, y_flow, y_torch, 1e-5, 1e-2)
    compare_result(test_case, x_flow_grad, x_torch_grad, 1e-5, 1e-2)

    print(f"============== PASSED =============")
    print("\n")


def _test_irfft(test_case, dtype=np.float32, params: dict = None):
    print(f"========== Start Testing ==========")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")

    x_flow, x_torch = tensor_builder(params=params, dtype=dtype)
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    print(f"irfft n: {n}")
    print(f"irfft dim: {dim}")
    print(f"irfft norm: {norm}")
    print(f"x_flow.dtype: {x_flow.dtype}")
    print("x_torch.dtype: ", x_torch.dtype)

    # forward
    y_torch = torch.fft.irfft(x_torch, n=n, dim=dim, norm=norm)
    y_torch_sum = y_torch.sum()

    # backward
    y_torch_sum.backward()

    # copy back to cpu memory
    x_torch_grad = x_torch.grad.detach().cpu()
    y_torch = y_torch.detach().cpu()


    # forward
    y_flow = flow._C.irfft(x_flow, n=n, dim=dim, norm=norm)
    y_flow_sum = y_flow.sum()

    # backward
    y_flow_sum.backward()

    # copy back to cpu memory
    x_flow_grad = x_flow.grad.detach().cpu()
    y_flow = y_flow.detach().cpu()
    if torch.is_conj(y_torch):
        y_torch = torch.resolve_conj(y_torch)
    if torch.is_conj(x_torch_grad):
        x_torch_grad = torch.resolve_conj(x_torch_grad)

    compare_result(test_case, y_flow, y_torch, 1e-5, 1e-2)
    compare_result(test_case, x_flow_grad, x_torch_grad, 1e-5, 1e-2)

    print(f"============== PASSED =============")
    print("\n")

def _test_hfft(test_case, dtype=np.complex64, params: dict = None):
    print(f"========== Start Testing ==========")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")

    x_flow, x_torch = tensor_builder(params=params, dtype=dtype)
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    print(f"hfft n: {n}")
    print(f"hfft dim: {dim}")
    print(f"hfft norm: {norm}")
    print(f"x_flow.dtype: {x_flow.dtype}")
    print("x_torch.dtype: ", x_torch.dtype)

    # forward
    y_torch = torch.fft.hfft(x_torch, n=n, dim=dim, norm=norm)
    y_torch_sum = y_torch.sum()

    # backward
    y_torch_sum.backward()

    # copy back to cpu memory
    x_torch_grad = x_torch.grad.detach().cpu()
    y_torch = y_torch.detach().cpu()

    # forward
    y_flow = flow._C.hfft(x_flow, n=n, dim=dim, norm=norm)

    y_flow_sum = y_flow.sum()

    # backward
    y_flow_sum.backward()

    # copy back to cpu memory
    x_flow_grad = x_flow.grad.detach().cpu()
    y_flow = y_flow.detach().cpu()
    if torch.is_conj(y_torch):
        y_torch = torch.resolve_conj(y_torch)
    if torch.is_conj(x_torch_grad):
        x_torch_grad = torch.resolve_conj(x_torch_grad)

    compare_result(test_case, y_flow, y_torch, 1e-5, 1e-2)
    compare_result(test_case, x_flow_grad, x_torch_grad, 1e-5, 1e-2)

    print(f"============== PASSED =============")
    print("\n")


def _test_ihfft(test_case, dtype=np.float32, params: dict = None):
    print(f"========== Start Testing ==========")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")

    x_flow, x_torch = tensor_builder(params=params, dtype=dtype)
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    print(f"ihfft n: {n}")
    print(f"ihfft dim: {dim}")
    print(f"ihfft norm: {norm}")
    print(f"x_flow.dtype: {x_flow.dtype}")
    print("x_torch.dtype: ", x_torch.dtype)

    # forward
    y_torch = torch.fft.ihfft(x_torch, n=n, dim=dim, norm=norm)
    y_torch_sum = y_torch.sum()

    # backward
    y_torch_sum.backward()

    # copy back to cpu memory
    x_torch_grad = x_torch.grad.detach().cpu()
    y_torch = y_torch.detach().cpu()

    # forward
    y_flow = flow._C.ihfft(x_flow, n=n, dim=dim, norm=norm)

    y_flow_sum = y_flow.sum()

    # backward
    y_flow_sum.backward()

    # copy back to cpu memory
    x_flow_grad = x_flow.grad.detach().cpu()
    y_flow = y_flow.detach().cpu()
    if torch.is_conj(y_torch):
        y_torch = torch.resolve_conj(y_torch)
    if torch.is_conj(x_torch_grad):
        x_torch_grad = torch.resolve_conj(x_torch_grad)

    compare_result(test_case, y_flow, y_torch, 1e-5, 1e-2)
    compare_result(test_case, x_flow_grad, x_torch_grad, 1e-5, 1e-2)

    print(f"============== PASSED =============")
    print("\n")

class TestFft(flow.unittest.TestCase):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.arg_dict["test_fun"] = [_test_fft, _test_ifft]
        test_case.arg_dict["dtype"] = [np.float32, np.float64, np.complex64, np.complex128]
    
    def test_gather(test_case):        
        test_case.arg_dict["params"] = []
        lower_n_dims = 1
        upper_n_dims = 5
        for _ in range(30):
            num_dims = np.random.randint(lower_n_dims, upper_n_dims)
            shape = [np.random.randint(1, 11) * 2 for _ in range(num_dims)]
            if np.random.randint(2) == 1:
                dim = np.random.randint(low=-num_dims, high=num_dims - 1)
            else:
                dim = -1

            norm = np.random.choice(["backward", "forward", "ortho", None])

            if np.random.randint(2) == 1:
                n = None
            else:
                n = np.random.randint(low=1, high=shape[dim] * 2)
                
            # shape = (12, 4, 10, 2)
            # n = 17
            # dim = 2
            # norm = None
                            
            test_case.arg_dict["params"].append(
                {"shape": shape, "n": n, "dim": dim, "norm": norm}
            )
        for arg in GenArgList(test_case.arg_dict):
            arg[0](test_case, *arg[1:])

class TestRFft(TestFft):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.arg_dict["test_fun"] = [_test_rfft]
        test_case.arg_dict["dtype"] = [np.float32, np.float64]

class TestIRFft(TestFft):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.arg_dict["test_fun"] = [_test_irfft]
        test_case.arg_dict["dtype"] = [np.complex64, np.complex128]

class TestHFft(TestFft):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.arg_dict["test_fun"] = [_test_hfft]
        test_case.arg_dict["dtype"] = [np.complex64, np.complex128]

class TestIHFft(TestFft):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.arg_dict["test_fun"] = [_test_ihfft]
        test_case.arg_dict["dtype"] = [np.float32, np.float64]

if __name__ == "__main__":
    unittest.main()
