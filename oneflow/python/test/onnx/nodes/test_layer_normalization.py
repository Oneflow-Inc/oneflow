import oneflow as flow
from util import convert_to_onnx_and_check


func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_ln(test_case):
    @flow.global_function(func_config)
    def ln(x=flow.FixedTensorDef((3, 4, 2, 5))):
        return flow.layers.layer_norm(x)

    convert_to_onnx_and_check(ln)
