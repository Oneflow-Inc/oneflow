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


import oneflow as flow
import torch
import numpy as np

import unittest
import oneflow.unittest
from numpy import random


@flow.unittest.skip_unless_1n1d()
class Test(flow.unittest.TestCase):
    def test_stft(test_case):
        rand_size = random.randint(513, 5000)
        np_tensor = random.rand(rand_size)
        rand_fft = random.randint(5, rand_size - 1)
        win_tensor = random.rand(rand_fft)

        x = torch.tensor(np_tensor).to("cuda")
        win = torch.tensor(win_tensor).to("cuda")
        y_torch = torch.stft(
            x,
            n_fft=rand_fft,
            window=win,
            win_length=rand_fft,
            center=True,
            onesided=True,
            normalized=False,
            return_complex=False,
        )

        x_flow = flow.tensor(np_tensor).to("cuda")
        win_flow = flow.tensor(win_tensor).to("cuda")

        y_flow = flow.stft(
            x_flow,
            n_fft=rand_fft,
            window=win_flow,
            win_length=rand_fft,
            center=True,
            onesided=True,
            normalized=False,
            return_complex=False,
        )

        test_case.assertTrue(
            np.allclose(y_flow.numpy(), y_torch.cpu().numpy(), rtol=1e-5, atol=1e-5)
        )

    # TODO(yzm):add test after support onesided,normalized,return_complex


if __name__ == "__main__":
    unittest.main()
