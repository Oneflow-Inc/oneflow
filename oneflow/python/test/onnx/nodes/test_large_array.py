import oneflow as flow
from util import convert_to_onnx_and_check


func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_large_array(test_case):
    @flow.global_function(func_config)
    def add_with_large_array():
        large_shape = (256 * 1024 * 1024 + 1,)
        x = flow.get_variable(
            name="x",
            shape=large_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        y = flow.get_variable(
            name="y",
            shape=large_shape,
            dtype=flow.float,
            initializer=flow.random_uniform_initializer(),
        )
        return flow.math.add_n([x, y])

    # ONNX Runtime optimizers doesn't support external data
    convert_to_onnx_and_check(
        add_with_large_array, external_data=True, ort_optimize=False
    )
