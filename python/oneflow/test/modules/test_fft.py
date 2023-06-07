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
import torch as torch_original
from packaging import version

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList

from oneflow.test_utils.automated_test_util import *


def is_cufft_available():
    if flow.cuda.is_available():
        (major, _minor) = flow.cuda.get_device_capability()
        return major >= 7
    else:
        return False


def is_complex_dtype(dtype):
    if hasattr(dtype, "pytorch") and hasattr(dtype, "oneflow"):
        # is DualObject
        return dtype.pytorch.is_complex
    else:
        return dtype in [
            flow.complex64,
            flow.complex128,
            torch_original.complex64,
            torch_original.complex128,
            torch.pytorch.complex64,
            torch.pytorch.complex128,
        ]


def gen_params_1d_fft(lower_n_dims=1, upper_n_dims=5):
    num_dims = np.random.randint(lower_n_dims, upper_n_dims)
    shape = [np.random.randint(1, 5) * 2 for _ in range(num_dims)]

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


def gen_params_2d_fft(lower_n_dims=2, upper_n_dims=5):
    num_dims = np.random.randint(lower_n_dims, upper_n_dims)
    shape = [np.random.randint(1, 5) * 2 for _ in range(num_dims)]
    len_fft_dim = np.random.randint(low=1, high=3)

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


def gen_params_nd_fft(lower_n_dims=2, upper_n_dims=5):
    num_dims = np.random.randint(lower_n_dims, upper_n_dims)
    shape = [np.random.randint(1, 5) * 2 for _ in range(num_dims)]
    len_fft_dim = np.random.randint(low=1, high=num_dims + 1)

    total_dims_range = np.arange(num_dims)
    if np.random.randint(2) == 1:
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


def _test_fft(test_case):

    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["1d"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["1d"]["upper_n_dims"]
    params = gen_params_1d_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]

    x = random_tensor(num_dims, dtype=float, *shape)
    if is_complex_dtype(x.dtype):
        # test fft_c2c
        dtype = test_case.dtype_dict["complex"]
        x = x.to(device=device, dtype=dtype)
    else:
        # test fft_r2c
        dtype = test_case.dtype_dict["real"]
        x = x.to(device=device, dtype=dtype)
    y = torch.fft.fft(x, n, dim, norm)
    return y


def _test_ifft(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["1d"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["1d"]["upper_n_dims"]
    params = gen_params_1d_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]

    x = random_tensor(num_dims, dtype=float, *shape)
    if is_complex_dtype(x.dtype):
        # test fft_c2c
        dtype = test_case.dtype_dict["complex"]
        x = x.to(device=device, dtype=dtype)
    else:
        # test fft_r2c
        dtype = test_case.dtype_dict["real"]
        x = x.to(device=device, dtype=dtype)

    y = torch.fft.ifft(x, n, dim, norm)

    return y


def _test_rfft(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["1d"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["1d"]["upper_n_dims"]
    params = gen_params_1d_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]

    dtype = test_case.dtype_dict["real"]

    x = random_tensor(num_dims, dtype=float, *shape).to(device=device, dtype=dtype)
    y = torch.fft.rfft(x, n, dim, norm)

    return y


def _test_irfft(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["1d"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["1d"]["upper_n_dims"]
    params = gen_params_1d_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    dtype = test_case.dtype_dict["complex"]

    x = random_tensor(num_dims, dtype=float, *shape).to(device=device, dtype=dtype)
    y = torch.fft.irfft(x, n, dim, norm)

    return y


def _test_hfft(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["1d"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["1d"]["upper_n_dims"]
    params = gen_params_1d_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    dtype = test_case.dtype_dict["complex"]

    x = random_tensor(num_dims, dtype=float, *shape).to(device=device, dtype=dtype)
    y = torch.fft.hfft(x, n, dim, norm)

    return y


def _test_ihfft(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["1d"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["1d"]["upper_n_dims"]
    params = gen_params_1d_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    dtype = test_case.dtype_dict["real"]

    x = random_tensor(num_dims, dtype=float, *shape).to(device=device, dtype=dtype)
    y = torch.fft.ihfft(x, n, dim, norm)

    return y


def _test_fft2(test_case):

    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["2d"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["2d"]["upper_n_dims"]
    params = gen_params_2d_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]

    x = random_tensor(num_dims, dtype=float, *shape)
    if is_complex_dtype(x.dtype):
        # test fft_c2c
        dtype = test_case.dtype_dict["complex"]
        x = x.to(device=device, dtype=dtype)
    else:
        # test fft_r2c
        dtype = test_case.dtype_dict["real"]
        x = x.to(device=device, dtype=dtype)
    y = torch.fft.fft2(x, n, dim, norm)

    return y


def _test_ifft2(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["2d"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["2d"]["upper_n_dims"]
    params = gen_params_2d_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]

    x = random_tensor(num_dims, dtype=float, *shape)
    if is_complex_dtype(x.dtype):
        # test fft_c2c
        dtype = test_case.dtype_dict["complex"]
        x = x.to(device=device, dtype=dtype)
    else:
        # test fft_r2c
        dtype = test_case.dtype_dict["real"]
        x = x.to(device=device, dtype=dtype)

    y = torch.fft.ifft2(x, n, dim, norm)

    return y


def _test_rfft2(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["2d"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["2d"]["upper_n_dims"]
    params = gen_params_2d_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]

    dtype = test_case.dtype_dict["real"]

    x = random_tensor(num_dims, dtype=float, *shape).to(device=device, dtype=dtype)
    y = torch.fft.rfft2(x, n, dim, norm)

    return y


def _test_irfft2(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["2d"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["2d"]["upper_n_dims"]
    params = gen_params_2d_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    dtype = test_case.dtype_dict["complex"]

    x = random_tensor(num_dims, dtype=float, *shape).to(device=device, dtype=dtype)
    y = torch.fft.irfft2(x, n, dim, norm)

    return y


def _test_hfft2(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["2d"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["2d"]["upper_n_dims"]
    params = gen_params_2d_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    dtype = test_case.dtype_dict["complex"]

    x = random_tensor(num_dims, dtype=float, *shape).to(device=device, dtype=dtype)
    y = torch.fft.hfft2(x, n, dim, norm)

    return y


def _test_ihfft2(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["2d"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["2d"]["upper_n_dims"]
    params = gen_params_2d_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    dtype = test_case.dtype_dict["real"]

    x = random_tensor(num_dims, dtype=float, *shape).to(device=device, dtype=dtype)
    y = torch.fft.ihfft2(x, n, dim, norm)

    return y


def _test_fftn(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["nd"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["nd"]["upper_n_dims"]
    params = gen_params_nd_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]

    x = random_tensor(num_dims, dtype=float, *shape)
    if is_complex_dtype(x.dtype):
        # test fft_c2c
        dtype = test_case.dtype_dict["complex"]
        x = x.to(device=device, dtype=dtype)
    else:
        # test fft_r2c
        dtype = test_case.dtype_dict["real"]
        x = x.to(device=device, dtype=dtype)
    y = torch.fft.fftn(x, n, dim, norm)

    return y


def _test_ifftn(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["nd"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["nd"]["upper_n_dims"]
    params = gen_params_nd_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]

    x = random_tensor(num_dims, dtype=float, *shape)
    if is_complex_dtype(x.dtype):
        # test fft_c2c
        dtype = test_case.dtype_dict["complex"]
        x = x.to(device=device, dtype=dtype)
    else:
        # test fft_r2c
        dtype = test_case.dtype_dict["real"]
        x = x.to(device=device, dtype=dtype)

    y = torch.fft.ifftn(x, n, dim, norm)

    return y


def _test_rfftn(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["nd"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["nd"]["upper_n_dims"]
    params = gen_params_nd_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]

    dtype = test_case.dtype_dict["real"]

    x = random_tensor(num_dims, dtype=float, *shape).to(device=device, dtype=dtype)
    y = torch.fft.rfftn(x, n, dim, norm)

    return y


def _test_irfftn(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["nd"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["nd"]["upper_n_dims"]
    params = gen_params_nd_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    dtype = test_case.dtype_dict["complex"]

    x = random_tensor(num_dims, dtype=float, *shape).to(device=device, dtype=dtype)
    y = torch.fft.irfftn(x, n, dim, norm)

    return y


def _test_hfftn(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["nd"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["nd"]["upper_n_dims"]
    params = gen_params_nd_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    dtype = test_case.dtype_dict["complex"]

    x = random_tensor(num_dims, dtype=float, *shape).to(device=device, dtype=dtype)
    y = torch.fft.hfftn(x, n, dim, norm)

    return y


def _test_ihfftn(test_case):
    if is_cufft_available():
        device = random_device()
    else:
        device = cpu_device()

    lower_n_dims = test_case.ndims_dict["nd"]["lower_n_dims"]
    upper_n_dims = test_case.ndims_dict["nd"]["upper_n_dims"]
    params = gen_params_nd_fft(lower_n_dims, upper_n_dims)

    num_dims = params["num_dims"]
    shape = params["shape"]
    n = params["n"]
    dim = params["dim"]
    norm = params["norm"]
    dtype = test_case.dtype_dict["real"]

    x = random_tensor(num_dims, dtype=float, *shape).to(device=device, dtype=dtype)
    y = torch.fft.ihfftn(x, n, dim, norm)

    return y


# NOTE: skip for multi-nodes and multi-devices now, because it failed in ci randomly
@flow.unittest.skip_unless_1n1d()
class TestComplex64Fft(flow.unittest.TestCase):
    def setUp(test_case):
        # should override by other data type of complex
        test_case.ndims_dict = {
            "1d": {"lower_n_dims": 1, "upper_n_dims": 5},
            "2d": {"lower_n_dims": 2, "upper_n_dims": 5},
            "nd": {"lower_n_dims": 1, "upper_n_dims": 5},
        }

        test_case.dtype_dict = {"real": torch.float32, "complex": torch.complex64}

        test_case.rtol = 1e-5
        test_case.atol = 1e-5
        test_case.initTestFft()

    def initTestFft(test_case):
        test_case.test_fft = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=True,
        )(_test_fft)

        test_case.test_ifft = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=True,
        )(_test_ifft)

        test_case.test_rfft = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=False,
        )(_test_rfft)

        test_case.test_irfft = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=True,
        )(_test_irfft)

        test_case.test_hfft = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=True,
        )(_test_hfft)

        test_case.test_ihfft = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=False,
        )(_test_ihfft)

        test_case.test_fft2 = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=True,
        )(_test_fft2)

        test_case.test_ifft2 = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=True,
        )(_test_ifft2)

        test_case.test_rfft2 = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=False,
        )(_test_rfft2)

        test_case.test_irfft2 = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol
            * 100,  # NOTE: ND-dimension of fft_c2r expands the numerical accuracy error
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=True,
        )(_test_irfft2)

        test_case.test_hfft2 = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol
            * 100,  # NOTE: ND-dimension of fft_c2r expands the numerical accuracy error
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=True,
        )(_test_hfft2)

        test_case.test_ihfft2 = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=False,
        )(_test_ihfft2)

        test_case.test_fftn = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol * 1e2,  # NOTE:
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=True,
        )(_test_fftn)

        test_case.test_ifftn = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol * 1e2,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=True,
        )(_test_ifftn)

        test_case.test_rfftn = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol * 1e2,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=False,
        )(_test_rfftn)

        test_case.test_irfftn = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol
            * 1e2,  # NOTE: ND-dimension of fft_c2r expands the numerical accuracy error
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=True,
        )(_test_irfftn)

        test_case.test_hfftn = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol
            * 1e2,  # NOTE: ND-dimension of fft_c2r expands the numerical accuracy error
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=True,
        )(_test_hfftn)

        test_case.test_ihfftn = autotest(
            n=5,
            auto_backward=True,
            rtol=test_case.rtol,
            atol=test_case.atol * 1e2,
            check_graph=False,
            check_grad_use_random_data=True,
            include_complex=False,
        )(_test_ihfftn)

    def test_1d_fft(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            test_case.test_fft,
            test_case.test_ifft,
            test_case.test_rfft,
            test_case.test_irfft,
            test_case.test_hfft,
            test_case.test_ihfft,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_2d_fft_except_hfft2(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            test_case.test_fft2,
            test_case.test_ifft2,
            test_case.test_rfft2,
            test_case.test_irfft2,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @unittest.skipIf(
        version.parse(torch_original.__version__) < version.parse("1.11.0"),
        "module 'torch.fft' has no attribute 'hfft2' or 'ihfft2' before '1.11.0'",
    )
    def test_2d_fft_hfft2(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [test_case.test_hfft2, test_case.test_ihfft2]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    def test_nd_fft_except_hfftn(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [
            test_case.test_fftn,
            test_case.test_ifftn,
            test_case.test_rfftn,
            test_case.test_irfftn,
        ]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

    @unittest.skipIf(
        version.parse(torch_original.__version__) < version.parse("1.11.0"),
        "module 'torch.fft' has no attribute 'hfftn' or 'ihfftn' before '1.11.0'",
    )
    def test_nd_fft_hfftn(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [test_case.test_hfftn, test_case.test_ihfftn]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


# NOTE: skip for multi-nodes and multi-devices now, because it failed in ci randomly
@flow.unittest.skip_unless_1n1d()
class TestComplex128Fft(TestComplex64Fft):
    def setUp(test_case):
        # should override by other data type of complex
        test_case.ndims_dict = {
            "1d": {"lower_n_dims": 1, "upper_n_dims": 5},
            "2d": {"lower_n_dims": 2, "upper_n_dims": 5},
            "nd": {"lower_n_dims": 1, "upper_n_dims": 5},
        }

        test_case.dtype_dict = {"real": torch.float64, "complex": torch.complex128}

        test_case.rtol = 1e-7
        test_case.atol = 1e-7
        test_case.initTestFft()


if __name__ == "__main__":
    unittest.main()
