import oneflow as flow
from util import convert_to_onnx_and_check

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_pad_float(test_case):
    @flow.function(func_config)
    def pad(x=flow.FixedTensorDef((3, 5))):
        return flow.pad(x, [(1, 2), (3, 4)], 1)

    convert_to_onnx_and_check(pad)
