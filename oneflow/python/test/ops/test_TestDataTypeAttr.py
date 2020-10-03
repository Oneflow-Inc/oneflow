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
from collections import OrderedDict

import numpy as np
import oneflow as flow
import oneflow.typing as oft
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type
import unittest
import os


def TestDataTypeAttr(input, output_type):
    assert output_type in flow.dtypes()
    return (
        flow.user_op_builder("TestDataTypeAttr")
        .Op("TestDataTypeAttr")
        .Input("in", [input])
        .Output("out")
        .Attr("output_type", output_type)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def RunTest(data_type):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def TestDataTypeAttrJob(input: oft.Numpy.Placeholder((10, 10), dtype=flow.float)):
        return TestDataTypeAttr(input, type_name_to_flow_type[data_type])

    input = np.random.random_sample((10, 10)).astype(np.float32)
    output = TestDataTypeAttrJob(input).get().numpy()
    assert output.dtype == type_name_to_np_type[data_type]


@flow.unittest.skip_unless_1n1d()
class Test_TestDataTypeAttr(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_data_type_attr(test_case):
        # TODO: fix bugs in ForeignOutputKernel with "float16" and "char" dtype, do not test these two dtypes here
        for data_type in ["float32", "double", "int8", "int32", "int64", "uint8"]:
            RunTest(data_type)


if __name__ == "__main__":
    unittest.main()
