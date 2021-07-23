import unittest
import os
from collections import OrderedDict
from oneflow.compatible import single_client as flow
from test_util import GenArgList

def distribute_reshape_test(device_type, device_num, input_shape, shape):
    assert device_type in ['gpu', 'cpu']
    flow.clear_default_session()
    flow.config.gpu_device_num(device_num)
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type='train', function_config=func_config)
    def ReshapeJob():
        with flow.scope.placement(device_type, '0:0-{}'.format(device_num - 1)):
            x = flow.get_variable('var_x', shape=input_shape, dtype=flow.float, initializer=flow.random_uniform_initializer(minval=2, maxval=5), trainable=True, distribute=flow.distribute.split(2))
            loss = flow.reshape(x, shape)
            flow.optimizer.SGD(flow.optimizer.PiecewiseConstantScheduler([], [0.0001]), momentum=0).minimize(loss)
            return (x, loss)
    (x, loss) = ReshapeJob().get()

@flow.unittest.skip_unless_1n2d()
class TestReshapeV2(flow.unittest.TestCase):

    def test_reshape(test_case):
        arg_dict = OrderedDict()
        arg_dict['device_type'] = ['gpu', 'cpu']
        arg_dict['device_num'] = [2]
        arg_dict['input_shape'] = [(5, 8, 16)]
        arg_dict['shape'] = [[-1, 16]]
        for arg in GenArgList(arg_dict):
            distribute_reshape_test(*arg)
if __name__ == '__main__':
    unittest.main()