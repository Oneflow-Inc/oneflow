import oneflow as flow
import numpy as np
from collections import OrderedDict

from test_util import GenArgList
from test_util import type_name_to_flow_type
from test_util import type_name_to_np_type


def TestListDataTypeAndListShapeAttr(input, out_shapes, out_types):
    assert isinstance(out_shapes, list)
    assert isinstance(out_types, list)
    return (
        flow.user_op_builder("TestListDataTypeAndListShapeAttr")
        .Op("TestListDataTypeAndListShapeAttr")
        .Input("in", [input])
        .Output("out", 3)
        .SetAttr("out_shapes", out_shapes, "AttrTypeListShape")
        .SetAttr("out_types", out_types, "AttrTypeListDataType")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )


def RunTest(out_shapes, out_types):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def TestListDataTypeAndListShapeAttrJob(input=flow.FixedTensorDef((10, 10), dtype=flow.float)):
        return TestListDataTypeAndListShapeAttr(
            input, out_shapes, [type_name_to_flow_type[data_type] for data_type in out_types]
        )

    input = np.random.random_sample((10, 10)).astype(np.float32)
    outputs = [x.ndarray() for x in TestListDataTypeAndListShapeAttrJob(input).get()]
    for i in range(len(outputs)):
        assert outputs[i].shape == out_shapes[i]
        assert outputs[i].dtype == type_name_to_np_type[out_types[i]]


def gen_arg_list():
    arg_dict = OrderedDict()
    arg_dict["out_shapes"] = [[(4, 4), (6, 6), (8, 8)]]
    # TODO: fix bugs in ForeignOutputKernel with "float16" and "char" dtype, do not test these two dtypes here
    arg_dict["out_types"] = [["float32", "double", "int8"], ["int32", "int64", "uint8"]]

    return GenArgList(arg_dict)


def test_data_type_attr(test_case):
    for arg in gen_arg_list():
        RunTest(*arg)
