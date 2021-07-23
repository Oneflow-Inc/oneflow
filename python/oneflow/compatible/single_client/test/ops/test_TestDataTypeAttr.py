import os
import unittest
from collections import OrderedDict

import numpy as np
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft


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
        for data_type in ["float32", "double", "int8", "int32", "int64", "uint8"]:
            RunTest(data_type)


if __name__ == "__main__":
    unittest.main()
