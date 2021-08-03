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


def make_job(a_shape, b_shape, trans_a=False, trans_b=False, xrts=[]):
    config = flow.FunctionConfig()
    xrt_util.set_xrt(config, xrts=xrts)
    @flow.global_function(function_config=config)
    def matmul_job(
        a:oft.Numpy.Placeholder(a_shape),
        b:oft.Numpy.Placeholder(b_shape)
    ):
        return flow.matmul(a, b, transpose_a=trans_a, transpose_b=trans_b)

    return matmul_job


class TestMatmul(unittest.TestCase):
    def make_shape(self, m, n, transpose):
        if transpose:
            return (n, m)
        else:
            return (m, n)

    def _test_body(self, a, b, trans_a, trans_b):
        func = make_job(a.shape, b.shape, trans_a=trans_a, trans_b=trans_b)
        out = func(a, b).get()
        out_np = out.numpy()
        flow.clear_default_session()
        for xrt in xrt_util.xrt_backends:
            xrt_job = make_job(a.shape, b.shape, trans_a=trans_a, trans_b=trans_b, xrts=[xrt])
            xrt_out = xrt_job(a, b).get()
            self.assertTrue(np.allclose(out_np, xrt_out.numpy(), rtol=0.001, atol=1e-05))
            flow.clear_default_session()

    def _test_ones_body(self, m, k, n, trans_a, trans_b, dtype=np.float32):
        shape_a = self.make_shape(m, k, trans_a)
        shape_b = self.make_shape(k, n, trans_b)
        a = np.ones(shape_a, dtype=dtype)
        b = np.ones(shape_b, dtype=dtype)
        self._test_body(a, b, trans_a, trans_b)

    def _test_random_body(self, m, k, n, trans_a, trans_b, dtype=np.float32):
        shape_a = self.make_shape(m, k, trans_a)
        shape_b = self.make_shape(k, n, trans_b)
        a = np.random.random(shape_a).astype(dtype)
        b = np.random.random(shape_b).astype(dtype)
        self._test_body(a, b, trans_a, trans_b)

    def test_ones1x1x1_input(self):
        print("run test_ones1x1x1_input: ")
        self._test_ones_body(1, 1, 1, False, False)
        self._test_ones_body(1, 1, 1, False, True)
        self._test_ones_body(1, 1, 1, True, False)
        self._test_ones_body(1, 1, 1, True, True)

    def test_random1x1x1_input(self):
        print("test_random1x1x1_input: ")
        self._test_random_body(1, 1, 1, False, False)
        self._test_random_body(1, 1, 1, False, True)
        self._test_random_body(1, 1, 1, True, False)
        self._test_random_body(1, 1, 1, True, True)

    def test_ones1x10x1_input(self):
        print("test_ones1x10x1_input: ")
        self._test_ones_body(1, 10, 1, False, False)
        self._test_ones_body(1, 10, 1, False, True)
        self._test_ones_body(1, 10, 1, True, False)
        self._test_ones_body(1, 10, 1, True, True)

    def test_random1x10x1_input(self):
        print("test_random1x10x1_input: ")
        self._test_random_body(1, 10, 1, False, False)
        self._test_random_body(1, 10, 1, False, True)
        self._test_random_body(1, 10, 1, True, False)
        self._test_random_body(1, 10, 1, True, True)

    def test_ones10x10x2_input(self):
        print("test_ones10x10x2_input: ")
        self._test_ones_body(10, 10, 2, False, False)
        self._test_ones_body(10, 10, 2, False, True)
        self._test_ones_body(10, 10, 2, True, False)
        self._test_ones_body(10, 10, 2, True, True)

    def test_random10x10x2_input(self):
        print("run test_random10x10x2_input: ")
        self._test_random_body(10, 10, 2, False, False)
        self._test_random_body(10, 10, 2, False, True)
        self._test_random_body(10, 10, 2, True, False)
        self._test_random_body(10, 10, 2, True, True)


if __name__ == "__main__":
    unittest.main()
