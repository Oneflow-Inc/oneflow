import oneflow as flow
from util import convert_to_onnx_and_check

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_transpose(test_case):
    @flow.global_function(func_config)
    def transpose(x=flow.FixedTensorDef((3, 5, 4))):
        return flow.transpose(x, perm=(2, 0, 1))

    convert_to_onnx_and_check(transpose)
