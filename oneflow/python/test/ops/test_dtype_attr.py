# Remove this file after review

import oneflow as flow
import numpy as np


def dtype_attr(input, output_type):
    return (
        flow.user_op_builder("DTypeAttr")
        .Op("dtype_attr")
        .Input("in", [input])
        .Output("out")
        .SetAttr("output_type", output_type, "AttrTypeDataType")
        .Build()
        .RemoteBlobList()[0]
    )


def test_dtyep_atrr(test_case):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.function(func_config)
    def TestDTypeAttrJob(input=flow.FixedTensorDef((10, 10), dtype=flow.float)):
        return dtype_attr(input, flow.int64)

    input = np.arange(100, dtype=np.float32).reshape((10, 10))
    output = TestDTypeAttrJob(input).get().ndarray()
    print(output.dtype)
