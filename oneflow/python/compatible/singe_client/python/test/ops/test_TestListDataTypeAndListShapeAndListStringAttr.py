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

import numpy as np
import oneflow as flow
import oneflow.typing as oft

from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


def TestListDataTypeAndListShapeAndListStringAttr(
    input, out_shapes, out_types, string_list
):
    assert isinstance(out_shapes, list)
    assert isinstance(out_types, list)
    return (
        flow.user_op_builder("TestListDataTypeAndListShapeAndListStringAttr")
        .Op("TestListDataTypeAndListShapeAndListStringAttr")
        .Input("in", [input])
        .Output("out", 3)
        .Attr("out_shapes", out_shapes)
        .Attr("out_types", out_types)
        .Attr("string_list", string_list)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )


def RunTest(out_shapes, out_types):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def TestListDataTypeAndListShapeAndListStringAttrJob(
        input: oft.Numpy.Placeholder((10, 10), dtype=flow.float)
    ):
        return TestListDataTypeAndListShapeAndListStringAttr(
            input,
            out_shapes,
            [type_name_to_flow_type[data_type] for data_type in out_types],
            ["string1", "string2", "string3"],
        )

    input = np.random.random_sample((10, 10)).astype(np.float32)
    outputs = [
        x.numpy() for x in TestListDataTypeAndListShapeAndListStringAttrJob(input).get()
    ]
    for i in range(len(outputs)):
        assert outputs[i].shape == out_shapes[i]
        assert outputs[i].dtype == type_name_to_np_type[out_types[i]]


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["out_shapes"] = [[(4, 4), (6, 6), (8, 8)]]
    # TODO: fix bugs in ForeignOutputKernel with "float16" and "char" dtype, do not test these two dtypes here
    arg_dict["out_types"] = [["float32", "double", "int8"], ["int32", "int64", "uint8"]]

    return GenArgList(arg_dict)


@flow.unittest.skip_unless_1n1d()
class Test_TestListDataTypeAndListShapeAndListStringAttr(flow.unittest.TestCase):
    def test_data_type_attr(test_case):
        for arg in gen_arg_list():
            RunTest(*arg)


if __name__ == "__main__":
    unittest.main()
