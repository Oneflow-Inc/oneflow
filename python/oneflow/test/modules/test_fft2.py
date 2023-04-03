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

    return x_flow, x_torch


def compare_result(test_case, a, b, rtol=1e-6, atol=1e-8):
    test_case.assertTrue(
        np.allclose(a.numpy(), b.numpy(), rtol=rtol, atol=atol),
        f"\na\n{a.numpy()}\n{'-' * 80}\nb:\n{b.numpy()}\n{'*' * 80}\ndiff:\n{a.numpy() - b.numpy()}",
    )


def _test_fft2(test_case, dtype=np.complex64, params: dict = None):
    print(f"========== Start Testing ==========")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")

    x_flow, x_torch = tensor_builder(params=params, dtype=dtype)
    n = params["n"]
    dims = params["dims"]
    norm = params["norm"]
    print(f"fftn n: {n}")
    print(f"fftn dims: {dims}")
    print(f"fftn norm: {norm}")
    print(f"x_flow.dtype: {x_flow.dtype}")
    print("x_torch.dtype: ", x_torch.dtype)

    # forward
    y_torch = torch.fft.fft2(x_torch, s=n, dim=dims, norm=norm)
    y_torch_sum = y_torch.sum()

    # backward
    y_torch_sum.backward()

    # copy back to cpu memory
    x_torch_grad = x_torch.grad.detach().cpu()
    y_torch = y_torch.detach().cpu()

    # forward
    y_flow = flow._C.fft2(x_flow, s=n, dim=dims, norm=norm)
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

    compare_result(test_case, y_flow, y_torch, 1e-6, 1e-2)
    compare_result(test_case, x_flow_grad, x_torch_grad, 1e-6, 1e-2)

    print(f"============== PASSED =============")
    print("\n")


def _test_ifft2(test_case, dtype=np.complex64, params: dict = None):
    print(f"========== Start Testing ==========")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")

    x_flow, x_torch = tensor_builder(params=params, dtype=dtype)
    n = params["n"]
    dims = params["dims"]
    norm = params["norm"]
    print(f"fftn n: {n}")
    print(f"fftn dims: {dims}")
    print(f"fftn norm: {norm}")
    print(f"x_flow.dtype: {x_flow.dtype}")
    print("x_torch.dtype: ", x_torch.dtype)

    # forward
    y_torch = torch.fft.ifft2(x_torch, s=n, dim=dims, norm=norm)
    y_torch_sum = y_torch.sum()

    # backward
    y_torch_sum.backward()

    # copy back to cpu memory
    x_torch_grad = x_torch.grad.detach().cpu()
    y_torch = y_torch.detach().cpu()

    # forward
    y_flow = flow._C.ifft2(x_flow, s=n, dim=dims, norm=norm)
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

    compare_result(test_case, y_flow, y_torch, 1e-6, 1e-2)
    compare_result(test_case, x_flow_grad, x_torch_grad, 1e-6, 1e-2)

    print(f"============== PASSED =============")
    print("\n")

def _test_rfft2(test_case, dtype=np.float32, params: dict = None):
    print(f"========== Start Testing ==========")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")

    x_flow, x_torch = tensor_builder(params=params, dtype=dtype)
    n = params["n"]
    dims = params["dims"]
    norm = params["norm"]
    print(f"rfftn n: {n}")
    print(f"rfftn dims: {dims}")
    print(f"rfftn norm: {norm}")
    print(f"x_flow.dtype: {x_flow.dtype}")
    print("x_torch.dtype: ", x_torch.dtype)

    # forward
    y_torch = torch.fft.rfft2(x_torch, s=n, dim=dims, norm=norm)
    y_torch_sum = y_torch.sum()

    # backward
    y_torch_sum.backward()

    # copy back to cpu memory
    x_torch_grad = x_torch.grad.detach().cpu()
    y_torch = y_torch.detach().cpu()

    # forward
    y_flow = flow._C.rfft2(x_flow, s=n, dim=dims, norm=norm)
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

    compare_result(test_case, y_flow, y_torch, 1e-6, 1e-2)
    compare_result(test_case, x_flow_grad, x_torch_grad, 1e-6, 1e-2)

    print(f"============== PASSED =============")
    print("\n")

def _test_irfft2(test_case, dtype=np.complex64, params: dict = None):
    print(f"========== Start Testing ==========")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")

    x_flow, x_torch = tensor_builder(params=params, dtype=dtype)
    n = params["n"]
    dims = params["dims"]
    norm = params["norm"]
    print(f"irfftn n: {n}")
    print(f"irfftn dims: {dims}")
    print(f"irfftn norm: {norm}")
    print(f"x_flow.dtype: {x_flow.dtype}")
    print("x_torch.dtype: ", x_torch.dtype)

    # forward
    y_torch = torch.fft.irfftn(x_torch, s=n, dim=dims, norm=norm)
    y_torch_sum = y_torch.sum()

    # backward
    y_torch_sum.backward()

    # copy back to cpu memory
    x_torch_grad = x_torch.grad.detach().cpu()
    y_torch = y_torch.detach().cpu()

    # forward
    y_flow = flow._C.irfft2(x_flow, s=n, dim=dims, norm=norm)
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

    compare_result(test_case, y_flow, y_torch, 1e-6, 1e-2)
    compare_result(test_case, x_flow_grad, x_torch_grad, 1e-6, 1e-2)

    print(f"============== PASSED =============")
    print("\n")

def _test_hfft2(test_case, dtype=np.complex64, params: dict = None):
    print(f"========== Start Testing ==========")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")

    x_flow, x_torch = tensor_builder(params=params, dtype=dtype)
    n = params["n"]
    dims = params["dims"]
    norm = params["norm"]
    print(f"irfftn n: {n}")
    print(f"irfftn dims: {dims}")
    print(f"irfftn norm: {norm}")
    print(f"x_flow.dtype: {x_flow.dtype}")
    print("x_torch.dtype: ", x_torch.dtype)

    # forward
    y_torch = torch.fft.hfft2(x_torch, s=n, dim=dims, norm=norm)
    y_torch_sum = y_torch.sum()

    # backward
    y_torch_sum.backward()

    # copy back to cpu memory
    x_torch_grad = x_torch.grad.detach().cpu()
    y_torch = y_torch.detach().cpu()

    # forward
    y_flow = flow._C.hfft2(x_flow, s=n, dim=dims, norm=norm)
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

    compare_result(test_case, y_flow, y_torch, 1e-6, 1e-2)
    compare_result(test_case, x_flow_grad, x_torch_grad, 1e-6, 1e-2)

    print(f"============== PASSED =============")
    print("\n")

def _test_ihfft2(test_case, dtype=np.float32, params: dict = None):
    print(f"========== Start Testing ==========")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")

    x_flow, x_torch = tensor_builder(params=params, dtype=dtype)
    n = params["n"]
    dims = params["dims"]
    norm = params["norm"]
    print(f"irfftn n: {n}")
    print(f"irfftn dims: {dims}")
    print(f"irfftn norm: {norm}")
    print(f"x_flow.dtype: {x_flow.dtype}")
    print("x_torch.dtype: ", x_torch.dtype)

    # forward
    y_torch = torch.fft.ihfft2(x_torch, s=n, dim=dims, norm=norm)
    y_torch_sum = y_torch.sum()

    # backward
    y_torch_sum.backward()

    # copy back to cpu memory
    x_torch_grad = x_torch.grad.detach().cpu()
    y_torch = y_torch.detach().cpu()

    # forward
    y_flow = flow._C.ihfft2(x_flow, s=n, dim=dims, norm=norm)
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

    compare_result(test_case, y_flow, y_torch, 1e-6, 1e-2)
    compare_result(test_case, x_flow_grad, x_torch_grad, 1e-6, 1e-2)

    print(f"============== PASSED =============")
    print("\n")

class TestFft2(flow.unittest.TestCase):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.arg_dict["test_fun"] = [_test_fft2, _test_ifft2]
        test_case.arg_dict["dtype"] = [np.float32, np.float64, np.complex64, np.complex128]
    
    def test_gather(test_case):
        # set up profiling functions
        test_case.arg_dict["params"] = []
        lower_n_dims = 2
        upper_n_dims = 5
        for _ in range(30):
            num_dims = np.random.randint(lower_n_dims, upper_n_dims)
            shape = [np.random.randint(1, 11) * 2 for _ in range(num_dims)]
            len_fft_dim = np.random.randint(low=1, high=num_dims + 1)

            total_dims_range = np.arange(num_dims)
            if np.random.randint(2) == 1:
                # dim = np.random.randint(low=-num_dims, high=num_dims-1)
                dims = np.random.choice(
                    total_dims_range, size=len_fft_dim, replace=False
                ).tolist()
            else:
                dims = (-2, -1)

            norm = np.random.choice(["backward", "forward", "ortho", None])
            len_fft_dim = len(dims)
            if np.random.randint(2) == 1 and dims is not None:
                n = []
                for i in range(len_fft_dim):
                    n_ = (
                        np.random.randint(low=1, high=2 * shape[i])
                        if np.random.randint(2) == 1
                        else -1
                    )
                    n.append(n_)
            else:
                n = None

            test_case.arg_dict["params"].append(
                {"shape": shape, "n": n, "dims": dims, "norm": norm}
            )

        for arg in GenArgList(test_case.arg_dict):
            arg[0](test_case, *arg[1:])

class TestRFft2(TestFft2):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.arg_dict["test_fun"] = [_test_rfft2]
        test_case.arg_dict["dtype"] = [np.float32, np.float64]
        
class TestIRFft2(TestFft2):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.arg_dict["test_fun"] = [_test_irfft2]
        test_case.arg_dict["dtype"] = [np.complex64, np.complex128]

class TestHFft2(TestFft2):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.arg_dict["test_fun"] = [_test_hfft2]
        test_case.arg_dict["dtype"] = [np.complex64, np.complex128]

class TestIHFft2(TestFft2):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.arg_dict["test_fun"] = [_test_ihfft2]
        test_case.arg_dict["dtype"] = [np.float32, np.float64]

if __name__ == "__main__":
    unittest.main()
