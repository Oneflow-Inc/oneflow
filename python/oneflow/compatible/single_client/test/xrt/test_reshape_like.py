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

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow
import oneflow.compatible.single_client.typing as oft

import xrt_util


def make_job(x_shape, like_shape, xrts=[]):
    config = flow.FunctionConfig()
    xrt_util.set_xrt(config, xrts=xrts)
    @flow.global_function(function_config=config)
    def reshape_like_job(
        x: oft.Numpy.Placeholder(x_shape),
        like: oft.Numpy.Placeholder(like_shape),
    ):
        return flow.reshape_like(x, like)

    return reshape_like_job

class TestReshapeLike(unittest.TestCase):
    def _test_body(self, x, like, dtype=np.float32):
        func = make_job(x.shape, like.shape)
        out = func(x, like).get()
        out_np = out.numpy()
        flow.clear_default_session()
        for xrt in xrt_util.xrt_backends:
            xrt_job = make_job(x.shape, like.shape, xrts=[xrt])
            xrt_out = xrt_job(x, like).get()
            self.assertTrue(np.allclose(out_np, xrt_out.numpy(), rtol=0.001, atol=1e-05))
            flow.clear_default_session()

    def _test_ones_body(self, x_shape, like_shape, dtype=np.float32):
        x = np.ones(x_shape, dtype=dtype)
        like = np.ones(like_shape, dtype=dtype)
        self._test_body(x, like, dtype=dtype)

    def _test_random_body(self, x_shape, like_shape, dtype=np.float32):
        x = np.random.random(x_shape).astype(dtype)
        like = np.random.random(like_shape).astype(dtype)
        self._test_body(x, like, dtype=dtype)

    def test_ones_input(self):
        self._test_ones_body((1, 10), (10,))
        self._test_ones_body((2, 10, 2), (4, 10))
        self._test_ones_body((2, 5, 2, 2), (2, 5, 4))

    def test_random_input(self):
        self._test_random_body((1, 10), (10,))
        self._test_random_body((2, 10, 2), (4, 10))
        self._test_random_body((2, 5, 2, 2), (2, 5, 4))


if __name__ == "__main__":
    unittest.main()
