import oneflow as flow
from util import convert_to_onnx_and_check


func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_reshape(test_case):
    @flow.function(func_config)
    def reshape(x=flow.FixedTensorDef((3, 4, 2, 5))):
        return flow.reshape(x, (4, 30))
    convert_to_onnx_and_check(reshape)



