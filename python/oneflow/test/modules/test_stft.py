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


class TestStft(flow.unittest.TestCase):
    @autotest(
        check_graph=False,
        check_grad_use_random_data=False,
        auto_backward=False,
    )
    def test_stft_with_1D_random_data(test_case):
        device = random_device()
        rand_size =  np.random.randint(10, 100000)
        rand_fft =  2 * np.random.randint(4, rand_size / 2)
        input_dims = [rand_size]
        win_dims = [rand_fft]
        x = random_tensor(1, *input_dims).to(device)
        win = random_tensor(1, *win_dims).to(device)
        y = torch.stft(
            x,
            n_fft=rand_fft,
            window=win,
            return_complex=False,
            onesided=True,
            center=True,
            normalized=True
        )
        return y


    def test_stft_with_2D_random_data(test_case):
        device = random_device()
        row_rand_size = np.random.randint(10, 500)
        col_rand_size = np.random.randint(10, 2000)
        rand_fft = 2 * np.random.randint(5, col_rand_size / 2)
        input_dims = [row_rand_size, col_rand_size]
        win_dims=[rand_fft]
        x = random_tensor(2, *input_dims).to(device)
        win = random_tensor(1, *win_dims).to(device)

        y = torch.stft(       
            x,
            n_fft=rand_fft,
            window=win,
            return_complex=False,
            onesided=True,
            center=True,
            normalized=True)
        return y
    
if __name__ == "__main__":
    unittest.main()
