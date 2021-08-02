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
    
def make_job(x_shape, y_shape, xrts=[]):
    config = flow.FunctionConfig()
    xrt_util.set_xrt(config, xrts=xrts)
    @flow.global_function(function_config=config)
    def add_job(
        x:oft.Numpy.Placeholder(x_shape),
        y:oft.Numpy.Placeholder(y_shape)
    ):
        return x + y + x

    return add_job


class TestAdd(unittest.TestCase):
    def _test_body(self, x, y):
        func = make_job(x.shape, y.shape)
        out = func(x, y).get()
        out_np = out.numpy()
        flow.clear_default_session()
        for xrt in xrt_util.xrt_backends:
            xrt_job = make_job(x.shape, y.shape, xrts=[xrt])
            xrt_out = xrt_job(x, y).get()
            self.assertTrue(np.allclose(out_np, xrt_out.numpy(), rtol=0.001, atol=1e-05))
            flow.clear_default_session()
       

    def _test_ones_body(self, x_shape, y_shape, dtype=np.float32):
        x = np.ones(x_shape, dtype=dtype)
        y = np.ones(y_shape, dtype=dtype)
        self._test_body(x, y,)

    def _test_random_body(self, x_shape, y_shape, dtype=np.float32):
        x = np.random.random(x_shape).astype(dtype)
        y = np.random.random(y_shape).astype(dtype)
        self._test_body(x, y)

    def test_ones_input(self):
        self._test_ones_body((1, 10), (1, 10))
        self._test_ones_body((2, 10, 2), (2, 10, 2))
        self._test_ones_body((2, 5, 2, 2), (2, 5, 2, 2))

    def test_random_input(self):
        self._test_random_body((1, 10), (1, 10))
        self._test_random_body((2, 10, 2), (2, 10, 2))
        self._test_random_body((2, 5, 2, 2), (2, 5, 2, 2))


if __name__ == "__main__":
    unittest.main()
