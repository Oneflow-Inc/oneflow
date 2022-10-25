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


# import oneflow as flow
# import torch

# import unittest
# import oneflow.unittest


from numpy import random

import unittest
from collections import OrderedDict

import numpy as np

import oneflow as flow
from oneflow.test_utils.test_util import GenArgList

from oneflow.test_utils.automated_test_util import *

# class TestStft(flow.unittest.TestCase):
#     @autotest(check_graph=False, check_grad_use_random_data=False, auto_backward=False)
#     def test_stft_with_1D_random_data(test_case):
#         device = random_device()
#         rand_size = 100#np.random.randint(10, 10000)
#         rand_fft =100# 2 * np.random.randint(4, rand_size / 2)
#         input_dims = [rand_size]
#         x = random_tensor(1, *input_dims).to(device)
#         y = torch.stft(x, n_fft=rand_fft, return_complex=False, onesided=True)
#         return y

    # def test_stft_with_2D_random_data(test_case):
    #     device = random_device()
    #     row_rand_size = np.random.randint(10, 1000)
    #     col_rand_size = np.random.randint(10, 1000)
    #     rand_fft = 2 * np.random.randint(5, col_rand_size / 2)
    #     input_dims = [row_rand_size, col_rand_size]
    #     x = random_tensor(2, *input_dims).to(device)
    #     y = torch.stft(x, n_fft=rand_fft, return_complex=False)
    #     return y


def test_stft_1D(test_case,device):
    rand_size = np.random.randint(10, 100000)
    np_tensor = np.random.rand(rand_size)
    rand_fft = np.random.randint(5, rand_size - 1)
    win_tensor = np.random.rand(rand_fft)

    x = torch.tensor(np_tensor).to(device)
    win = torch.tensor(win_tensor).to(device)
        # print(x.shape)
        # print(rand_fft)
    # y_torch = torch.stft(
    #         x,
    #         n_fft=rand_fft,
    #         window=win,
    #         win_length=rand_fft,
    #         center=True,
    #         onesided=True,
    #         normalized=True,
    #         return_complex=False,
    #     )
    y_torch=torch.tensor([1,2])
    x_flow = flow.tensor(np_tensor).to(device)
    win_flow = flow.tensor(win_tensor).to(device)

    y_flow = flow.stft(
            x_flow,
            n_fft=rand_fft,
            window=win_flow,
            win_length=rand_fft,
            center=True,
            onesided=True,
            normalized=True,
            return_complex=False,
        )
    print( np.array_equal(y_flow.numpy(), y_torch.cpu().numpy()))
    test_case.assertTrue(
            np.array_equal(y_flow.numpy(), y_torch.cpu().numpy())
        )

#     def test_stft_2D(test_case):
#         rand_length =1# np.random.randint(20, 150)
#         rand_winth =200# np.random.randint(513, 2000)
#         np_tensor = np.random.rand(rand_length, rand_winth)
#         # BUG(yzm):n_fft is not aligned with pytorch when it is odd
#         rand_fft =51# 2 * np.random.randint(5, rand_winth / 2 - 1)
#         win_tensor = np.random.rand(rand_fft)

#         x = torch.tensor(np_tensor).to("cuda")
#         win = torch.tensor(win_tensor).to("cuda")
#         y_torch = torch.stft(
#             x,
#             n_fft=rand_fft,
#             window=win,
#             win_length=rand_fft,
#             center=False,
#             onesided=True,
#             normalized=True,
#             return_complex=False,
#         )
#         y_torch=torch.randn(10)
#         x_flow = flow.tensor(np_tensor).to("cuda")
#         win_flow = flow.tensor(win_tensor).to("cuda")

#         y_flow = flow.stft(
#             x_flow,
#             n_fft=rand_fft,
#             window=win_flow,
#             win_length=rand_fft,
#             center=False,
#             onesided=True,
#             normalized=True,
#             return_complex=False,
#         )
#         print(y_flow.numpy())
#         print(y_torch.cpu().numpy())
#         test_case.assertTrue(
#             np.allclose(y_flow.numpy(), y_torch.cpu().numpy(), rtol=1e-5, atol=1e-5)
#         )

        

# TODO(yzm):add test after support onesided,normalized,return_complex


@flow.unittest.skip_unless_1n1d()
class TestStft(flow.unittest.TestCase):
    def test_stft(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [test_stft_1D]
        arg_dict["device"] = ["cpu", "cuda"]
        for arg in GenArgList(arg_dict):
            arg[0](test_case, *arg[1:])


if __name__ == "__main__":
    unittest.main()
