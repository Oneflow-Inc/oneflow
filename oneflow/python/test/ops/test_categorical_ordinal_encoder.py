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
import sys
import numpy as np
import oneflow as flow
import oneflow.typing as oft
import typing
import unittest
import os


def _test_categorical_ordinal_encoder(
    test_case, device_tag, dtype, size, capacity, num_tokens, num_iters
):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())

    @flow.global_function(function_config=func_config)
    def test_job(
        x: oft.Numpy.Placeholder(shape=(size,), dtype=dtype)
    ) -> typing.Tuple[oft.Numpy, oft.Numpy]:
        with flow.scope.placement(device_tag, "0:0"):
            y = flow.layers.categorical_ordinal_encoder(x, capacity=capacity)
            z = flow.layers.categorical_ordinal_encoder(
                x, capacity=capacity, name="encode1"
            )
            # z = flow.layers.categorical_ordinal_encoder(x, capacity=320)
            return y, z

    tokens = np.random.randint(-sys.maxsize, sys.maxsize, size=[num_tokens]).astype(
        flow.convert_oneflow_dtype_to_numpy_dtype(dtype)
    )
    k_set = set()
    v_set = set()
    kv_set = set()
    vk_set = set()

    for i in range(num_iters):
        x = tokens[np.random.randint(0, num_tokens, (size,))]
        y, z = test_job(x)

        test_case.assertEqual(x.shape, y.shape)
        if device_tag == "cpu":
            test_case.assertTrue(
                np.array_equal(y, z),
                "\ny: {}\n{}\nz: {}\n{}".format(y.shape, y, z.shape, z),
            )

        for k, v in zip(x, y):
            k_set.add(k)
            v_set.add(v)
            kv_set.add((k, v))
            vk_set.add((v, k))

    unique_size = len(k_set)
    test_case.assertEqual(len(v_set), unique_size)
    test_case.assertEqual(len(kv_set), unique_size)
    test_case.assertEqual(len(vk_set), unique_size)


@flow.unittest.skip_unless_1n1d()
class TestCategoricalOrdinalEncoder(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_categorical_ordinal_encoder_gpu_large(test_case):
        _test_categorical_ordinal_encoder(
            test_case=test_case,
            device_tag="gpu",
            dtype=flow.int64,
            size=10000,
            capacity=320000,
            num_tokens=200000,
            num_iters=256,
        )

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_categorical_ordinal_encoder_gpu_small(test_case):
        _test_categorical_ordinal_encoder(
            test_case=test_case,
            device_tag="gpu",
            dtype=flow.int32,
            size=10,
            capacity=250,
            num_tokens=200,
            num_iters=4,
        )

    def test_categorical_ordinal_encoder_cpu_large(test_case):
        _test_categorical_ordinal_encoder(
            test_case=test_case,
            device_tag="cpu",
            dtype=flow.int64,
            size=20000,
            capacity=220000,
            num_tokens=200000,
            num_iters=100,
        )

    def test_categorical_ordinal_encoder_cpu_very_large(test_case):
        _test_categorical_ordinal_encoder(
            test_case=test_case,
            device_tag="cpu",
            dtype=flow.int64,
            size=50000,
            capacity=1000000,
            num_tokens=500000,
            num_iters=100,
        )


if __name__ == "__main__":
    unittest.main()
