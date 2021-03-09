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
from __future__ import absolute_import

import os
import unittest

import numpy as np
import oneflow as flow
import oneflow.typing as oft

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def multi_square_sum(
    x, name=None,
):

    return (
        flow.user_op_builder(name if name is not None else "MultiSquareSum")
        .Op("multi_square_sum")
        .Input("x", x)
        .Output("y")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def _check(test_case, xs, y):
    ref_y = np.sum(np.array([np.sum(x ** 2) for x in xs]))
    test_case.assertTrue(np.allclose(y, ref_y))


def _run_test(test_case, x, n, dtype, device):
    flow.clear_default_session()

    @flow.global_function(function_config=func_config)
    def multi_square_sum_job(x: oft.Numpy.Placeholder(x.shape, dtype=dtype)):
        with flow.scope.placement(device, "0:0"):
            xs = [x + 0.1 * i for i in range(n)]
            return multi_square_sum(xs)

    y = multi_square_sum_job(x).get()
    _check(test_case, [(x + 0.1 * i).astype(np.float32) for i in range(n)], y.numpy())


@flow.unittest.skip_unless_1n1d()
class TestMultiSquareSum(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_multi_square_sum_random_gpu(test_case):
        x = np.random.rand(3, 4, 5).astype(np.float32)
        _run_test(test_case, x, 5, flow.float32, "gpu")
        _run_test(test_case, x, 5, flow.float32, "gpu")
        _run_test(test_case, x, 88, flow.float32, "gpu")
        _run_test(test_case, x, 64, flow.float32, "gpu")

    def test_multi_square_sum_random_cpu(test_case):
        x = np.random.rand(3, 4, 5).astype(np.float32)
        _run_test(test_case, x, 5, flow.float32, "cpu")


if __name__ == "__main__":
    unittest.main()
