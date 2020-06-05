import oneflow as flow
from util import convert_to_onnx_and_check


func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_bn_nchw(test_case):
    @flow.function(func_config)
    def bn(x=flow.FixedTensorDef((3, 4, 2, 5))):
        params_shape = (4,)
        mean = flow.get_variable(name='mean', shape=params_shape,
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        variance = flow.get_variable(name='var', shape=params_shape,
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        gamma = flow.get_variable(name='gamma', shape=params_shape,
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        beta = flow.get_variable(name='beta', shape=params_shape,
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        return flow.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-5, axis=1)
    convert_to_onnx_and_check(bn)


def test_bn_nhwc(test_case):
    @flow.function(func_config)
    def bn(x=flow.FixedTensorDef((3, 4, 2, 5))):
        params_shape = (5,)
        mean = flow.get_variable(name='mean', shape=params_shape,
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        variance = flow.get_variable(name='var', shape=params_shape,
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        gamma = flow.get_variable(name='gamma', shape=params_shape,
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        beta = flow.get_variable(name='beta', shape=params_shape,
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        return flow.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-5, axis=-1)
    convert_to_onnx_and_check(bn)

