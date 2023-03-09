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

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList

def tensor_builder(params: dict, dtype=flow.float32):
    input_shape = params["shape"]
    
    # generate random input
    x = np.random.randn(*input_shape)
    other = np.random.randn(*input_shape)
    
    # transfer to gpu memory
    tensor_x = flow.FloatTensor(x).to(dtype=dtype, device="cuda").requires_grad_(True)
    tensor_other = flow.FloatTensor(other).to(dtype=dtype, device="cuda").requires_grad_(True)
    
    return tensor_x, tensor_other

def compare_result(test_case, a, b, rtol=1e-5, atol=1e-8):
    test_case.assertTrue(
        np.allclose(a.numpy(), b.numpy(), rtol=rtol, atol=atol),
        f"\na\n{a.numpy()}\n{'-' * 80}\nb:\n{b.numpy()}\n{'*' * 80}\ndiff:\n{a.numpy() - b.numpy()}",
    )
    
def _test_fft(test_case, params: dict, dtype=flow.float32):
    print(f"========== Start Testing ==========")
    print(f"weight tensor: merged")
    print(f"tensor shape: {params['shape']}")
    print(f"dtype: {dtype}")
    
    x, other = tensor_builder(params=params, dtype=dtype)
    
    # forward
    y = x * other
    
    # backward
    y.sum().backward()
    
    # copy back to cpu memory
    x_grad = x.grad.detach().cpu()
    other_grad = other.grad.detach().cpu()
    y = y.detach().cpu()
    
    
    fft_x = x.detach().clone().requires_grad_(True)
    fft_other = other.detach().clone().requires_grad_(True)
    
    # forward
    fft_y = flow._C.fft(
        x=fft_x, other=fft_other
    )
    
    # backward
    fft_y.sum().backward()

    # copy back to cpu memory
    fft_x_grad = fft_x.grad.detach().cpu()
    fft_other_grad = fft_other.grad.detach().cpu()
    fft_y = fft_y.detach().cpu()

    compare_result(test_case, fft_y, y, 1e-5, 1e-2)
    compare_result(test_case, fft_x_grad, x_grad, 1e-5, 1e-2)
    compare_result(test_case, fft_other_grad, other_grad, 1e-5, 1e-2)
    
    print(f"============== PASSED =============")
    print("\n")


class TestFft(flow.unittest.TestCase):
    def test_gather(test_case):
        arg_dict = OrderedDict()
        # set up test functions
        arg_dict["test_fun"] = [
            _test_fft,
        ]

        # set up profiling functions        
        arg_dict["params"] = []
        for _ in range(10):
            num_dims = np.random.randint(1, 4)
            shape = [np.random.randint(1,11) * 8 for _ in range(num_dims)]
            arg_dict["params"].append({"shape" : shape})
        
        arg_dict["dtype"] = [flow.float32, flow.float64]

        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])

if __name__ == "__main__":
    unittest.main()