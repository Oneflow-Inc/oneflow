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
import numpy as np
import oneflow as flow
import oneflow.typing as oft
import unittest
import os

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def _check_unique(test_case, x, y, idx, count, num_unique):
    ref_y, ref_count = np.unique(x, return_counts=True)
    sorted_idx = np.argsort(ref_y)
    ref_y = ref_y[sorted_idx]
    ref_count = ref_count[sorted_idx]
    num_unique = num_unique.item()
    test_case.assertTrue(num_unique, np.size(ref_y))
    y = y[0:num_unique]
    test_case.assertTrue(np.array_equal(y[idx], x))
    sorted_idx = np.argsort(y)
    test_case.assertTrue(np.array_equal(ref_y, y[sorted_idx]))
    count = count[0:num_unique]
    test_case.assertTrue(np.array_equal(count[sorted_idx], ref_count))


def _run_test(test_case, x, dtype, device):
    @flow.global_function(function_config=func_config)
    def UniqueWithCountsJob(x: oft.Numpy.Placeholder(x.shape, dtype=dtype)):
        with flow.scope.placement(device, "0:0"):
            return flow.experimental.unique_with_counts(x)

    y, idx, count, num_unique = UniqueWithCountsJob(x).get()
    _check_unique(
        test_case, x, y.numpy(), idx.numpy(), count.numpy(), num_unique.numpy()
    )


@flow.unittest.skip_unless_1n1d()
class TestUnique(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_unique_with_counts_int(test_case):
        x = np.asarray(list(range(32)) * 2).astype(np.int32)
        np.random.shuffle(x)
        _run_test(test_case, x, flow.int32, "gpu")

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_unique_with_counts_float(test_case):
        x = np.asarray(list(range(32)) * 2).astype(np.float32)
        np.random.shuffle(x)
        _run_test(test_case, x, flow.float32, "gpu")

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_unique_with_counts_random_gpu(test_case):
        x = np.random.randint(0, 32, 1024).astype(np.int32)
        np.random.shuffle(x)
        _run_test(test_case, x, flow.int32, "gpu")

    def test_unique_with_counts_random_cpu(test_case):
        x = np.random.randint(0, 32, 1024).astype(np.int32)
        np.random.shuffle(x)
        _run_test(test_case, x, flow.int32, "cpu")


if __name__ == "__main__":
    unittest.main()
