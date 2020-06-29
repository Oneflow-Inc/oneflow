import oneflow as flow
from util import convert_to_onnx_and_check


func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_bias_add_nchw(test_case):
    @flow.global_function(func_config)
    def bias_add_nchw(x=flow.FixedTensorDef((3, 4, 2, 5))):
        y = flow.get_variable(
            name="y",
            shape=(4,),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.nn.bias_add(x, y, "NCHW")

    convert_to_onnx_and_check(bias_add_nchw)


def test_bias_add_nhwc(test_case):
    @flow.global_function(func_config)
    def bias_add_nhwc(x=flow.FixedTensorDef((3, 4, 2, 5))):
        y = flow.get_variable(
            name="y",
            shape=(5,),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.nn.bias_add(x, y, "NHWC")

    convert_to_onnx_and_check(bias_add_nhwc)
