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
import oneflow.experimental as flow


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestBroadCastLike(flow.unittest.TestCase):
    def test_broadcast_like(test_case):
        input = flow.Tensor(
            np.ones(shape=(3, 1, 1), dtype=np.float32), dtype=flow.float32
        )
        like_tensor = flow.Tensor(
            np.ones(shape=(3, 3, 3), dtype=np.float32), dtype=flow.float32
        )
        of_out = flow.broadcast_like(input, like_tensor, broadcast_axes=(1, 2))
        np_out = np.ones(shape=(3, 3, 3))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_broadcast_like_3dim(test_case):
        input = flow.Tensor(
            np.ones(shape=(1, 3, 2), dtype=np.float32), dtype=flow.float32
        )
        like_tensor = flow.Tensor(
            np.ones(shape=(3, 3, 2), dtype=np.float32), dtype=flow.float32
        )
        of_out = flow.broadcast_like(input, like_tensor, broadcast_axes=(0,))
        np_out = np.ones(shape=(3, 3, 2))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))

    def test_broadcast_like_4dim(test_case):
        input = flow.Tensor(
            np.ones(shape=(1, 3, 2, 1), dtype=np.float32), dtype=flow.float32
        )
        like_tensor = flow.Tensor(
            np.ones(shape=(3, 3, 2, 3), dtype=np.float32), dtype=flow.float32
        )
        of_out = flow.broadcast_like(input, like_tensor, broadcast_axes=(0, 3))
        np_out = np.ones(shape=(3, 3, 2, 3))
        test_case.assertTrue(np.allclose(of_out.numpy(), np_out))


if __name__ == "__main__":
    unittest.main()
