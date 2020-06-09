import oneflow as flow
from util import convert_to_onnx_and_check


func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_matmul(test_case):
    @flow.function(func_config)
    def matmul():
        a = flow.get_variable(name='a', shape=(2, 3),
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        b = flow.get_variable(name='b', shape=(3, 4),
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        return flow.matmul(a, b)
    convert_to_onnx_and_check(matmul)


def test_matmul_ta(test_case):
    @flow.function(func_config)
    def matmul():
        a = flow.get_variable(name='a', shape=(3, 2),
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        b = flow.get_variable(name='b', shape=(3, 4),
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        return flow.matmul(a, b, transpose_a=True)
    convert_to_onnx_and_check(matmul)


def test_matmul_tb(test_case):
    @flow.function(func_config)
    def matmul():
        a = flow.get_variable(name='a', shape=(2, 3),
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        b = flow.get_variable(name='b', shape=(4, 3),
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        return flow.matmul(a, b, transpose_b=True)
    convert_to_onnx_and_check(matmul)



def test_matmul_ta_tb(test_case):
    @flow.function(func_config)
    def matmul():
        a = flow.get_variable(name='a', shape=(3, 2),
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        b = flow.get_variable(name='b', shape=(4, 3),
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        return flow.matmul(a, b, transpose_a=True, transpose_b=True)
    convert_to_onnx_and_check(matmul)


def test_batch_matmul(test_case):
    @flow.function(func_config)
    def matmul():
        a = flow.get_variable(name='a', shape=(4, 2, 3),
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        b = flow.get_variable(name='b', shape=(4, 3, 4),
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        return flow.matmul(a, b)
    convert_to_onnx_and_check(matmul)


