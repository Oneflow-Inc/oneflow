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
import os
import unittest
import numpy as np
import oneflow as flow


def _test(test_case, device_num):
    m, k, n = 5, 6, 7
    a_shape = (m, k)
    b_shape = (k, n)
    c_shape = (n,)

    flow.config.gpu_device_num(device_num)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float32)
    # func_config.prune_parallel_cast_ops(True)

    @flow.global_function("train", function_config=func_config)
    def test_fn(
        a: flow.typing.Numpy.Placeholder(a_shape),
        b: flow.typing.Numpy.Placeholder(b_shape),
        c: flow.typing.Numpy.Placeholder(c_shape),
    ) -> flow.typing.Numpy:
        # print(f"a.split_axis: {a.split_axis}")
        # print(f"b.split_axis: {b.split_axis}")
        # print(f"c.split_axis: {c.split_axis}")
        var_a = flow.get_variable(
            name="var_a",
            shape=a_shape,
            dtype=flow.float32,
            initializer=flow.ones_initializer(),
            distribute=flow.distribute.split(1),
        )
        # S0 -> S1
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(1)"])
        a = var_a * a
        out = flow.matmul(a, b)
        # P -> B
        out = flow.hierarchical_parallel_cast(out, parallel_distribution=["B"])
        # S0 -> B
        c = flow.hierarchical_parallel_cast(c, parallel_distribution=["B"])
        out = flow.nn.bias_add(out, c)
        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
        flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(out)
        return out

    a = np.random.rand(*a_shape).astype(np.float32)
    b = np.random.rand(*b_shape).astype(np.float32)
    c = np.random.rand(*c_shape).astype(np.float32)
    out = test_fn(a, b, c)
    test_case.assertTrue(np.allclose(out, np.matmul(a, b) + c))


@flow.unittest.skip_unless_1n2d()
class TestParallelCast(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_on_gpu(test_case):
        _test(test_case, 2)


if __name__ == "__main__":
    unittest.main()
