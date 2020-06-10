import oneflow as flow
from util import convert_to_onnx_and_check
import oneflow.core.common.data_type_pb2 as data_type_conf_util

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def test_gather_nd(test_case):
    @flow.function(func_config)
    def gather_nd():
        x = flow.get_variable(name='x', shape=(2, 3, 4),
                               dtype=flow.float, initializer=flow.random_uniform_initializer())
        y = flow.get_variable(name='y', shape=(2, 3),
                               dtype=flow.int64, initializer=flow.random_uniform_initializer(0, 1, data_type_conf_util.kInt64))
        return flow.gather_nd(x, y)
    convert_to_onnx_and_check(gather_nd)

