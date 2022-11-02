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
import oneflow as flow
import oneflow.unittest
import numpy as np


@flow.unittest.skip_unless_1n1d()
class TestModule(flow.unittest.TestCase):
    def test_stft_illegal_input_dim(test_case):
        np_tensor = np.arange(1, 13, dtype=float).reshape(2, 2, 3)

        with test_case.assertRaises(RuntimeError) as ctx:
            x_flow = flow.tensor(np_tensor)
            flow.stft(
                x_flow,
                n_fft=4,
                center=True,
                onesided=True,
                return_complex=False,
                normalized=False,
            )
        test_case.assertTrue("Expected a 1D or 2D tensor,but got" in str(ctx.exception))

    def test_stft_illegal_nfft(test_case):
        np_tensor = np.arange(1, 13, dtype=float).reshape(4, 3)
        win_tensor = np.arange(1, 5, dtype=float)

        with test_case.assertRaises(RuntimeError) as ctx:
            x_flow = flow.tensor(np_tensor)
            flow_win = flow.tensor(win_tensor)

            flow.stft(
                x_flow,
                n_fft=-1,
                window=flow_win,
                center=True,
                onesided=True,
                return_complex=False,
                normalized=False,
            )
        test_case.assertTrue("Expected 0 < n_fft <" in str(ctx.exception))

    def test_stft_illegal_hop_length(test_case):
        np_tensor = np.arange(1, 13, dtype=float).reshape(4, 3)

        with test_case.assertRaises(RuntimeError) as ctx:
            x_flow = flow.tensor(np_tensor)

            flow.stft(
                x_flow,
                n_fft=4,
                hop_length=-1,
                center=True,
                onesided=True,
                return_complex=False,
                normalized=False,
            )
        test_case.assertTrue("Expected hop_length > 0, but got" in str(ctx.exception))

    def test_stft_illegal_win_length(test_case):
        np_tensor = np.arange(1, 13, dtype=float).reshape(4, 3)

        with test_case.assertRaises(RuntimeError) as ctx:
            x_flow = flow.tensor(np_tensor)

            flow.stft(
                x_flow,
                n_fft=4,
                win_length=-1,
                center=True,
                onesided=True,
                return_complex=False,
                normalized=False,
            )
        test_case.assertTrue(
            "Expected 0 < win_length <=n_fft ,but got" in str(ctx.exception)
        )

    def test_stft_illegal_window(test_case):
        np_tensor = np.arange(1, 13, dtype=float).reshape(2, 6)
        win_tensor = np.arange(1, 10, dtype=float)

        with test_case.assertRaises(RuntimeError) as ctx:
            x_flow = flow.tensor(np_tensor)
            flow_win = flow.tensor(win_tensor)

            flow.stft(
                x_flow,
                n_fft=4,
                window=flow_win,
                center=True,
                onesided=True,
                return_complex=False,
                normalized=False,
            )
        test_case.assertTrue(
            "Expected a 1D window tensor of size equal to win_length="
            in str(ctx.exception)
        )


if __name__ == "__main__":
    unittest.main()
