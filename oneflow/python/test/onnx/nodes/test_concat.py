import oneflow as flow
from util import convert_to_onnx_and_check

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_concat_axis0(test_case):
    @flow.function(func_config)
    def concat():
        variables = []
        for i in range(4):
            variables.append(flow.get_variable(name=str(i), shape=(2, 3), dtype=flow.float, initializer=flow.random_uniform_initializer()))
        return flow.concat(variables, axis=0)
    convert_to_onnx_and_check(concat)


def test_concat_axis1(test_case):
    @flow.function(func_config)
    def concat():
        variables = []
        for i in range(4):
            variables.append(flow.get_variable(name=str(i), shape=(2, 3), dtype=flow.float, initializer=flow.random_uniform_initializer()))
        return flow.concat(variables, axis=1)
    convert_to_onnx_and_check(concat)

