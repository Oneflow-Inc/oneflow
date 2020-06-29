import oneflow as flow
from util import convert_to_onnx_and_check


func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_relu(test_case):
    @flow.global_function(func_config)
    def relu(x=flow.FixedTensorDef((3, 4, 2, 5))):
        return flow.math.relu(x)

    convert_to_onnx_and_check(relu)
