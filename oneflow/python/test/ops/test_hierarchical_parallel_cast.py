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
import oneflow as flow


def _test(test_case):
    flow.config.gpu_device_num(4)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)

    @flow.global_function("predict", function_config=func_config)
    def test_fn(x: flow.typing.Numpy.Placeholder((1024, 1024)),) -> flow.typing.Numpy:
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[2, 2], parallel_distribution=["S(0)", "S(0)"],
        )
        v = flow.get_variable(
            name="v",
            shape=(1024, 1024),
            parallel_hierarchy=(2, 2),
            parallel_distribution=["B", "B"],
            initializer=flow.ones_initializer(),
        )
        x = flow.matmul(x, v)
        x = flow.math.relu(x)
        x = flow.hierarchical_parallel_cast(
            x, parallel_hierarchy=[4], parallel_distribution=["S(0)"]
        )
        return x

    x_arr = np.random.rand(1024, 1024).astype(np.float32)
    y_arr = test_fn(x_arr).numpy()
    test_case.assertTrue(np.allclose(y_arr, x_arr))


@flow.unittest.skip_unless_1n4d()
class TestHierarchicalParallelCast(flow.unittest.TestCase):
    def test_hierarchy_parallel_cast(test_case):
        _test(test_case)


if __name__ == "__main__":
    unittest.main()
