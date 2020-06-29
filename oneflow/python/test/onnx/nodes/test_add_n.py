import oneflow as flow
from util import convert_to_onnx_and_check


func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_add_2(test_case):
    @flow.function(func_config)
    def add_2():
        x = flow.get_variable(
            name="x",
            shape=(2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        y = flow.get_variable(
            name="y",
            shape=(2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.math.add_n([x, y])

    convert_to_onnx_and_check(add_2)


def test_add_3(test_case):
    @flow.function(func_config)
    def add_3():
        x = flow.get_variable(
            name="x",
            shape=(2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        y = flow.get_variable(
            name="y",
            shape=(2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        z = flow.get_variable(
            name="z",
            shape=(2, 3),
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.math.add_n([x, y, z])

    convert_to_onnx_and_check(add_3)


def test_add_many(test_case):
    @flow.function(func_config)
    def add_many():
        variables = []
        for i in range(50):
            variables.append(
                flow.get_variable(
                    name=str(i),
                    shape=(2, 3),
                    dtype=flow.float,
                    initializer=flow.random_uniform_initializer(),
                )
            )
        return flow.math.add_n(variables)

    convert_to_onnx_and_check(add_many)
