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
import oneflow.typing as tp
import numpy as np


@flow.unittest.skip_unless_1n2d()
class TestDemoMatmul(flow.unittest.TestCase):
    def test_watch(test_case):
        flow.config.gpu_device_num(2)
        flow.config.enable_debug_mode(True)

        expected = np.array(
            [[30, 30, 30, 30], [30, 30, 30, 30], [30, 30, 30, 30], [30, 30, 30, 30],]
        ).astype(np.float32)

        def Watch(x: tp.Numpy):
            test_case.assertTrue(np.allclose(x, expected))

        @flow.global_function()
        def Matmul(
            x: tp.Numpy.Placeholder((4, 4), dtype=flow.float32),
            y: tp.Numpy.Placeholder((4, 4), dtype=flow.float32),
        ) -> tp.Numpy:
            s = flow.matmul(x, y)  # model parallel
            flow.watch(s, Watch)
            z = flow.matmul(s, x)  # data parallel
            return z

        x = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],]).astype(
            np.float32
        )

        y = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4],]).astype(
            np.float32
        )
        Matmul(x, y)


if __name__ == "__main__":
    unittest.main()
