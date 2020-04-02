import oneflow as flow
import numpy as np
from collections import OrderedDict

from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type


def TestDataTypeAttr(input, output_type):
    return (
        flow.user_op_builder("TestDataTypeAttr")
        .Op("TestDataTypeAttr")
        .Input("in", [input])
        .Output("out")
        .SetAttr("output_type", output_type, "AttrTypeDataType")
        .Build()
        .RemoteBlobList()[0]
    )


def RunTest(shape, data_type):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def TestDataTypeAttrJob(input=flow.FixedTensorDef(shape, dtype=flow.float)):
        return TestDataTypeAttr(input, type_name_to_flow_type[data_type])

    input = np.random.random_sample((shape)).astype(np.float32)
    output = TestDataTypeAttrJob(input).get().ndarray()
    assert output.dtype == type_name_to_np_type[data_type]


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["shape"] = [(10, 10)]
    # TODO: fix bugs in ForeignOutputKernel with "float" and "char" dtype, do not test these two dtypes here
    arg_dict["data_type"] = ["float32", "double", "int8", "int32", "int64", "uint8"]

    return GenArgList(arg_dict)


def test_data_type_attr(test_case):
    for arg in gen_arg_list():
        RunTest(*arg)
