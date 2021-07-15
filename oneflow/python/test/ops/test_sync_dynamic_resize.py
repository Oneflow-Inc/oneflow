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
from collections import OrderedDict

import os
import numpy as np
import oneflow as flow
import oneflow.typing as oft
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


@flow.unittest.skip_unless_1n1d()
class TestSyncDynamicResize(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_sync_dynamic_resize(_):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu", "cpu"]
        arg_dict["x_shape"] = [
            (100,),
            (1000, 10),
        ]
        arg_dict["data_type"] = ["float32", "double", "int32", "int64"]
        arg_dict["size_type"] = ["int32", "int64"]

        for device_type, x_shape, data_type, size_type in GenArgList(arg_dict):
            flow.clear_default_session()
            func_config = flow.FunctionConfig()
            func_config.default_data_type(flow.float)

            @flow.global_function(function_config=func_config)
            def TestJob(
                x: oft.Numpy.Placeholder(
                    x_shape, dtype=type_name_to_flow_type[data_type]
                ),
                size: oft.Numpy.Placeholder(
                    (1,), dtype=type_name_to_flow_type[size_type]
                ),
            ):
                with flow.scope.placement(device_type, "0:0"):
                    return flow.sync_dynamic_resize(x, size)

            size = np.random.randint(0, x_shape[0])
            x = np.random.rand(*x_shape).astype(type_name_to_np_type[data_type])
            y = (
                TestJob(x, np.array([size]).astype(type_name_to_np_type[size_type]))
                .get()
                .numpy_list()[0]
            )
            assert np.array_equal(y, x[:size])


if __name__ == "__main__":
    unittest.main()
