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

import unittest
from collections import OrderedDict

import numpy as np
import re

import oneflow as flow
from oneflow.test_utils.test_util import GenArgList

from oneflow.test_utils.automated_test_util import *


def getRandBoolvalue():
    value = np.random.randint(0, 2)
    if value == 1:
        return True
    else:
        return False


def getRandFFtvalue():
    pow = np.random.randint(2, 5)
    result = 1
    for i in range(pow):
        result = result * 2
    return result


def is_cufft_available():
    if flow.cuda.is_available():
        (major, _minor) = flow.cuda.get_device_capability()
        return major >= 7
    else:
        return False


class TestStft(flow.unittest.TestCase):
    @autotest(
        n=20, check_graph=False, check_grad_use_random_data=False, auto_backward=False,
    )
    def test_stft_with_1D_random_data(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()
        rand_fft = getRandFFtvalue()
        rand_size = np.random.randint(rand_fft, 300)
        input_dims = [rand_size]
        win_dims = [rand_fft]
        x = random_tensor(1, *input_dims).to(device)
        win = random_tensor(1, *win_dims).to(device)
        onesided_value = getRandBoolvalue()
        center_value = getRandBoolvalue()
        normalized_value = getRandBoolvalue()
        y = torch.stft(
            x,
            n_fft=rand_fft,
            window=win,
            return_complex=False,
            onesided=onesided_value,
            center=center_value,
            normalized=normalized_value,
        )
        return y

    def test_stft_with_2D_random_data(test_case):
        if is_cufft_available():
            device = random_device()
        else:
            device = cpu_device()
        row_rand_size = np.random.randint(1, 50)
        rand_fft = getRandFFtvalue()
        col_rand_size = np.random.randint(rand_fft, 300)
        input_dims = [row_rand_size, col_rand_size]
        win_dims = [rand_fft]
        x = random_tensor(2, *input_dims).to(device)
        win = random_tensor(1, *win_dims).to(device)
        onesided_value = getRandBoolvalue()
        center_value = getRandBoolvalue()
        normalized_value = getRandBoolvalue()
        y = torch.stft(
            x,
            n_fft=rand_fft,
            window=win,
            return_complex=False,
            onesided=onesided_value,
            center=center_value,
            normalized=normalized_value,
        )
        return y


if __name__ == "__main__":
    unittest.main()
