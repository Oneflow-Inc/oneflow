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


def _check(test_case, x_indices, x_values, y_indices, y_values, num_unique):
    ref_indices = np.unique(x_indices)
    np.sort(ref_indices)
    num_unique = num_unique.item()
    test_case.assertTrue(num_unique == ref_indices.shape[0])
    key_to_idx = dict(zip(ref_indices, range(num_unique)))
    ref_values = np.zeros((num_unique, y_values.shape[-1]), y_values.dtype)
    for i in range(x_indices.shape[0]):
        ref_values[key_to_idx[x_indices[i].item()]] += x_values[i]
    y_indices = y_indices[0:num_unique]
    y_values = y_values[0:num_unique]
    sorted_idx = np.argsort(y_indices)
    y_indices = y_indices[sorted_idx]
    y_values = y_values[sorted_idx]
    test_case.assertTrue(np.array_equal(ref_indices, y_indices))
    test_case.assertTrue(np.allclose(ref_values, y_values))


def _run_test(test_case, indices, values, indices_dtype, values_dtype, device):
    @flow.global_function(function_config=func_config)
    def TestJob(
        indices: oft.Numpy.Placeholder(indices.shape, dtype=indices_dtype),
        values: oft.Numpy.Placeholder(values.shape, dtype=values_dtype),
    ):
        with flow.scope.placement(device, "0:0"):
            return flow.experimental.indexed_slices_reduce_sum(indices, values)

    out_indices, out_values, num_unique = TestJob(indices, values).get()
    _check(
        test_case,
        indices,
        values,
        out_indices.numpy(),
        out_values.numpy(),
        num_unique.numpy(),
    )


@flow.unittest.skip_unless_1n1d()
class TestIndexedSlicesReduceSum(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_indexed_slices_reduce_sum_gpu(test_case):
        indices = np.random.randint(0, 32, 1024).astype(np.int32)
        values = np.random.rand(1024, 8).astype(np.float32)
        _run_test(test_case, indices, values, flow.int32, flow.float32, "gpu")

    def test_indexed_slices_reduce_sum_cpu(test_case):
        indices = np.random.randint(0, 32, 1024).astype(np.int32)
        values = np.random.rand(1024, 8).astype(np.float32)
        _run_test(test_case, indices, values, flow.int32, flow.float32, "cpu")


if __name__ == "__main__":
    unittest.main()
