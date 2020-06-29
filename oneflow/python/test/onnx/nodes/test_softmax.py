import oneflow as flow
from util import convert_to_onnx_and_check

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_softmax(test_case):
    @flow.function(func_config)
    def softmax(x=flow.FixedTensorDef((3, 5))):
        return flow.nn.softmax(x)

    convert_to_onnx_and_check(softmax)


def test_softmax_with_axis(test_case):
    @flow.function(func_config)
    def softmax(x=flow.FixedTensorDef((3, 5, 4))):
        return flow.nn.softmax(x, axis=1)

    convert_to_onnx_and_check(softmax)
