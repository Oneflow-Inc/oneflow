import unittest
import numpy as np
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft
from collections import OrderedDict
from test_util import Args, GenArgDict
func_config = flow.FunctionConfig()
func_config.default_logical_view(flow.scope.mirrored_view())
func_config.default_data_type(flow.float)

def test_repeat_acc(test_case, device_type, shape, dtype, acc_num):
    flow.clear_default_session()
    if flow.eager_execution_enabled():
        return

    @flow.global_function(function_config=func_config)
    def RepeatAccJob(a: oft.Numpy.Placeholder(shape)):
        if dtype == 'float16':
            return flow.cast(flow.acc(flow.repeat(flow.cast(a, flow.float16), acc_num), acc_num), flow.float)
        else:
            return flow.acc(flow.repeat(a, acc_num), acc_num)
    x = np.random.rand(*shape).astype(np.float32)
    y = RepeatAccJob(x).get().numpy()
    z = x * acc_num
    if dtype == 'float16':
        z = x.astype(np.float16) * acc_num
        z = z.astype(np.float32)
    test_case.assertTrue(np.allclose(y, z, rtol=1e-05, atol=1e-05))

@flow.unittest.skip_unless_1n1d()
class TestRepeatAcc(flow.unittest.TestCase):

    def test_case(test_case):
        arg_dict = OrderedDict()
        arg_dict['device_type'] = ['gpu', 'cpu']
        arg_dict['shape'] = [(1024, 1024, 4)]
        arg_dict['dtype'] = ['float16', 'float32', 'double']
        arg_dict['acc_num'] = [5]
        for arg in GenArgDict(arg_dict):
            if arg['device_type'] == 'cpu' and arg['dtype'] == 'float16':
                continue
            test_repeat_acc(test_case, **arg)
if __name__ == '__main__':
    unittest.main()