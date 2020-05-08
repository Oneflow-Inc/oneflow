import oneflow as flow
import numpy as np
from collections import OrderedDict

from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type


def TestDataTypeAttr(input, output_type):
    assert output_type in flow.dtypes
    return (
        flow.user_op_builder("TestDataTypeAttr")
        .Op("TestDataTypeAttr")
        .Input("in", [input])
        .Output("out")
        .SetAttr("output_type", output_type, "AttrTypeDataType")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def RunTest(data_type):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def TestDataTypeAttrJob(input=flow.FixedTensorDef((10, 10), dtype=flow.float)):
        return TestDataTypeAttr(input, type_name_to_flow_type[data_type])

    input = np.random.random_sample((10, 10)).astype(np.float32)
    output = TestDataTypeAttrJob(input).get().ndarray()
    assert output.dtype == type_name_to_np_type[data_type]


def test_data_type_attr(test_case):
    # TODO: fix bugs in ForeignOutputKernel with "float16" and "char" dtype, do not test these two dtypes here
    for data_type in ["float32", "double", "int8", "int32", "int64", "uint8"]:
        RunTest(data_type)
