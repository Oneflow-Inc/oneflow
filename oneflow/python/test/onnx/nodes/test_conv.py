import oneflow as flow
from util import convert_to_onnx_and_check

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_conv2d_k2d1_valid(test_case):
    @flow.function(func_config)
    def conv2d_k3s1_valid(x=flow.FixedTensorDef((2, 4, 3, 5))):
        return flow.layers.conv2d(x, 6, kernel_size=3, strides=1, padding="VALID")
    convert_to_onnx_and_check(conv2d_k3s1_valid)

def test_conv2d_s2_valid(test_case):
    @flow.function(func_config)
    def conv2d_s2_valid(x=flow.FixedTensorDef((2, 4, 3, 5))):
        return flow.layers.conv2d(x, 6, kernel_size=1, strides=2, padding="VALID")
    convert_to_onnx_and_check(conv2d_s2_valid)

def test_conv2d_s2_same(test_case):
    @flow.function(func_config)
    def conv2d_s2_same(x=flow.FixedTensorDef((2, 4, 3, 5))):
        return flow.layers.conv2d(x, 6, kernel_size=3, strides=2, padding="SAME")
    convert_to_onnx_and_check(conv2d_s2_same)

def test_conv2d_k3s1_nhwc_valid(test_case):
    @flow.function(func_config)
    def conv2d_k3s1_nhwc_valid(x=flow.FixedTensorDef((2, 3, 5, 4))):
        return flow.layers.conv2d(x, 6, kernel_size=3, strides=1, padding="VALID", data_format='NHWC')

    convert_to_onnx_and_check(conv2d_k3s1_nhwc_valid)

