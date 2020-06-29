import oneflow as flow
from util import convert_to_onnx_and_check

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_max_pooling_2d_k3s1_valid_nhwc(test_case):
    @flow.global_function(func_config)
    def max_pooling_2d_k3s1_valid_nhwc(x=flow.FixedTensorDef((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.max_pool2d(
            x, ksize=3, strides=1, padding="VALID", data_format="NHWC"
        )

    convert_to_onnx_and_check(max_pooling_2d_k3s1_valid_nhwc)


def test_max_pooling_2d_k3s1_same_nhwc(test_case):
    @flow.global_function(func_config)
    def max_pooling_2d_k3s1_same_nhwc(x=flow.FixedTensorDef((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.max_pool2d(
            x, ksize=3, strides=1, padding="SAME", data_format="NHWC"
        )

    convert_to_onnx_and_check(max_pooling_2d_k3s1_same_nhwc)


def test_max_pooling_2d_k2s2_same_nhwc(test_case):
    @flow.global_function(func_config)
    def max_pooling_2d_k2s2_same_nhwc(x=flow.FixedTensorDef((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.max_pool2d(
            x, ksize=2, strides=2, padding="SAME", data_format="NHWC"
        )

    convert_to_onnx_and_check(max_pooling_2d_k2s2_same_nhwc)


def test_max_pooling_2d_k3s1_valid_nchw(test_case):
    @flow.global_function(func_config)
    def max_pooling_2d_k3s1_valid_nchw(x=flow.FixedTensorDef((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.max_pool2d(
            x, ksize=3, strides=1, padding="VALID", data_format="NCHW"
        )

    convert_to_onnx_and_check(max_pooling_2d_k3s1_valid_nchw)


def test_avg_pooling_2d_k3s1_valid_nhwc(test_case):
    @flow.global_function(func_config)
    def avg_pooling_2d_k3s1_valid_nhwc(x=flow.FixedTensorDef((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.avg_pool2d(
            x, ksize=3, strides=1, padding="VALID", data_format="NHWC"
        )

    convert_to_onnx_and_check(avg_pooling_2d_k3s1_valid_nhwc)


def test_avg_pooling_2d_k3s1_same_nhwc(test_case):
    @flow.global_function(func_config)
    def avg_pooling_2d_k3s1_same_nhwc(x=flow.FixedTensorDef((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.avg_pool2d(
            x, ksize=3, strides=1, padding="SAME", data_format="NHWC"
        )

    convert_to_onnx_and_check(avg_pooling_2d_k3s1_same_nhwc)


def test_avg_pooling_2d_k2s2_same_nhwc(test_case):
    @flow.global_function(func_config)
    def avg_pooling_2d_k2s2_same_nhwc(x=flow.FixedTensorDef((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.avg_pool2d(
            x, ksize=2, strides=2, padding="SAME", data_format="NHWC"
        )

    convert_to_onnx_and_check(avg_pooling_2d_k2s2_same_nhwc)


def test_avg_pooling_2d_k3s1_valid_nchw(test_case):
    @flow.global_function(func_config)
    def avg_pooling_2d_k3s1_valid_nchw(x=flow.FixedTensorDef((2, 3, 5, 4))):
        x += flow.get_variable(
            name="v1",
            shape=(1, 1),
            dtype=flow.float,
            initializer=flow.zeros_initializer(),
        )
        return flow.nn.avg_pool2d(
            x, ksize=3, strides=1, padding="VALID", data_format="NCHW"
        )

    convert_to_onnx_and_check(avg_pooling_2d_k3s1_valid_nchw)
