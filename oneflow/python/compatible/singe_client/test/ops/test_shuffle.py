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
import uuid
from collections import OrderedDict

import numpy as np
import oneflow as flow
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type

import oneflow.typing as oft


@flow.unittest.skip_unless_1n1d()
class TestShuffle(flow.unittest.TestCase):
    def test_shuffle(_):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["x_shape"] = [
            (100,),
            (10, 1000),
            (10, 10, 2000),
        ]
        arg_dict["data_type"] = ["float32", "double", "int32", "int64"]

        for device_type, x_shape, data_type in GenArgList(arg_dict):
            assert device_type in ["gpu", "cpu"]
            assert data_type in ["float32", "double", "int8", "int32", "int64"]
            flow.clear_default_session()
            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)

            @flow.global_function(function_config=flow.FunctionConfig())
            def TestJob(
                x: oft.Numpy.Placeholder(
                    x_shape, dtype=type_name_to_flow_type[data_type]
                )
            ):
                with flow.scope.placement(device_type, "0:0"):
                    return flow.random.shuffle(x)

            x = np.random.randn(*x_shape).astype(type_name_to_np_type[data_type])
            ret = TestJob(x).get().numpy()
            assert np.array_equal(x, ret) == False, x_shape
            x.sort(0)
            ret.sort(0)
            assert np.array_equal(x, ret), x_shape

            assert device_type in ["gpu", "cpu"]
            assert data_type in ["float32", "double", "int8", "int32", "int64"]
            flow.clear_default_session()
            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)

            @flow.global_function(function_config=flow.FunctionConfig())
            def TestJob1(
                x: oft.Numpy.Placeholder(
                    x_shape, dtype=type_name_to_flow_type[data_type]
                )
            ):
                with flow.scope.placement(device_type, "0:0"):
                    return flow.random.generate_random_batch_permutation_indices(x)

            x = np.random.randn(*x_shape).astype(type_name_to_np_type[data_type])
            ret = TestJob1(x).get().numpy()
            idx = np.arange(x_shape[0]).astype(np.int32)
            assert np.array_equal(idx, ret) == False, x_shape
            idx.sort()
            ret.sort()
            assert np.array_equal(idx, ret), x_shape


if __name__ == "__main__":
    unittest.main()
