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
from numpy import random
import torch
import unittest
from collections import OrderedDict

import numpy as np
import re

import oneflow as flow
from oneflow.test_utils.test_util import GenArgList

from oneflow.test_utils.automated_test_util import *


def is_cufft_available():
    if flow.cuda.is_available():
        (major, _minor) = flow.cuda.get_device_capability()
        return major >= 7
    else:
        return False


def is_complex_dtype(dtype):
    if dtype in [flow.complex64, flow.complex128, torch.complex64, torch.complex128]:
        return True
    return False


class Test1DFft(flow.unittest.TestCase):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.lower_n_dims = 1
        test_case.upper_n_dims = 5

        test_case.dtype_list = [
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]

    def gen_params(test_case):
        num_dims = np.random.randint(test_case.lower_n_dims, test_case.upper_n_dims + 1)
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

        params = {
            "num_dims": num_dims,
            "shape": shape,
            "n": n,
            "dim": dim,
            "norm": norm,
        }
        return params

    @autotest(
        n=40,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-5,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_fft(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(0, 4)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.fft(x, n, dim, norm)

        return y

    @autotest(
        n=40,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-5,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_ifft(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(0, 4)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.ifft(x, n, dim, norm)

        return y

    @autotest(
        n=20,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-5,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_rfft(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(0, 2)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.rfft(x, n, dim, norm)

        return y

    @autotest(
        n=20,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-5,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_irfft(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(2, 4)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.irfft(x, n, dim, norm)

        return y

    @autotest(
        n=20,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-5,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_hfft(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(2, 4)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.hfft(x, n, dim, norm)

        return y

    @autotest(
        n=20,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-5,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_ihfft(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(0, 2)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.ihfft(x, n, dim, norm)

        return y


class Test2DFft(flow.unittest.TestCase):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.lower_n_dims = 2
        test_case.upper_n_dims = 5

        test_case.dtype_list = [
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]

    def gen_params(test_case):
        num_dims = np.random.randint(test_case.lower_n_dims, test_case.upper_n_dims)
        shape = [np.random.randint(1, 11) * 2 for _ in range(num_dims)]
        len_fft_dim = np.random.randint(low=1, high=num_dims + 1)

        total_dims_range = np.arange(num_dims)
        if np.random.randint(2) == 1:
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

        params = {
            "num_dims": num_dims,
            "shape": shape,
            "n": n,
            "dim": dims,
            "norm": norm,
        }
        return params

    @autotest(
        n=40,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-3,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_fft2(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(0, 4)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.fft2(x, n, dim, norm)

        return y

    @autotest(
        n=40,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-3,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_ifft2(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(0, 4)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.ifft2(x, n, dim, norm)

        return y

    @autotest(
        n=20,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-3,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_rfft2(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(0, 2)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.rfft2(x, n, dim, norm)

        return y

    @autotest(
        n=20,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-3,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_irfft2(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(2, 4)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.irfft2(x, n, dim, norm)

        return y

    @autotest(
        n=20,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-3,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_hfft2(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(2, 4)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.hfft2(x, n, dim, norm)

        return y

    @autotest(
        n=20,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-3,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_ihfft2(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(0, 2)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.ihfft2(x, n, dim, norm)

        return y


class TestNDFft(flow.unittest.TestCase):
    def setUp(test_case):
        test_case.arg_dict = OrderedDict()
        test_case.lower_n_dims = 1
        test_case.upper_n_dims = 5

        test_case.dtype_list = [
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]

    def gen_params(test_case):
        num_dims = np.random.randint(test_case.lower_n_dims, test_case.upper_n_dims)
        shape = [np.random.randint(1, 11) * 2 for _ in range(num_dims)]
        len_fft_dim = np.random.randint(low=1, high=num_dims + 1)

        total_dims_range = np.arange(num_dims)
        if np.random.randint(2) == 1:
            # dim = np.random.randint(low=-num_dims, high=num_dims-1)
            dims = np.random.choice(
                total_dims_range, size=len_fft_dim, replace=False
            ).tolist()
        else:
            dims = None

        norm = np.random.choice(["backward", "forward", "ortho", None])

        if np.random.randint(2) == 1:
            n = None
        else:
            n = []
            len_fft_dim = (
                len(dims)
                if dims is not None
                else np.random.randint(low=1, high=num_dims + 1)
            )
            for i in range(len_fft_dim):
                n_ = (
                    np.random.randint(low=1, high=2 * shape[i])
                    if np.random.randint(2) == 1
                    else -1
                )
                n.append(n_)

        params = {
            "num_dims": num_dims,
            "shape": shape,
            "n": n,
            "dim": dims,
            "norm": norm,
        }
        return params

    @autotest(
        n=40,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-3,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_fftn(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(0, 4)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.fftn(x, n, dim, norm)

        return y

    @autotest(
        n=40,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-3,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_ifftn(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(0, 4)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.ifftn(x, n, dim, norm)

        return y

    @autotest(
        n=20,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-3,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_rfftn(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(0, 2)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.rfftn(x, n, dim, norm)

        return y

    @autotest(
        n=20,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-3,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_irfftn(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(2, 4)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.irfftn(x, n, dim, norm)

        return y

    @autotest(
        n=20,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-3,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_hfftn(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(2, 4)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.hfftn(x, n, dim, norm)

        return y

    @autotest(
        n=20,
        auto_backward=True,
        rtol=1e-5,
        atol=1e-3,
        check_graph=False,
        check_grad_use_random_data=False,
    )
    def test_ihfftn(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()

        params = test_case.gen_params()
        print(params)
        num_dims = params["num_dims"]
        shape = params["shape"]
        n = params["n"]
        dim = params["dim"]
        norm = params["norm"]
        dtype = test_case.dtype_list[np.random.randint(0, 2)]

        if is_complex_dtype(dtype):
            x = random_tensor(num_dims, dtype=complex, *shape).to(
                device=device, dtype=dtype
            )
        else:
            x = random_tensor(num_dims, dtype=float, *shape).to(
                device=device, dtype=dtype
            )
        print(x.dtype)
        y = torch.fft.ihfftn(x, n, dim, norm)

        return y


if __name__ == "__main__":
    unittest.main()
