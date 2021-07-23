import unittest
import numpy as np
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft

@unittest.skipIf(True, 'skip for now because of single-client tensor_list removed')
class TestDynamicReshape(flow.unittest.TestCase):

    def test_dynamic_reshape(test_case):
        data_shape = (10, 10, 10)
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(type='train', function_config=func_config)
        def DynamicReshapeJob(x: oft.ListNumpy.Placeholder(data_shape)):
            reshape_out1 = flow.reshape(x, (-1, 20))
            my_model = flow.get_variable('my_model', shape=(20, 32), dtype=flow.float, initializer=flow.random_uniform_initializer(minval=-10, maxval=10), trainable=True)
            my_model = flow.cast_to_current_logical_view(my_model)
            mm_out = flow.matmul(reshape_out1, my_model)
            reshape_out2 = flow.reshape(mm_out, (-1, 8, 4))
            flow.optimizer.SGD(flow.optimizer.PiecewiseConstantScheduler([], [0.0001]), momentum=0).minimize(reshape_out2)
            return reshape_out1
        data = [np.random.rand(*data_shape).astype(np.float32) for i in range(2)]
        out = DynamicReshapeJob(data).get().numpy_list()
        for i in range(2):
            test_case.assertTrue(np.array_equal(np.reshape(data[i], (50, 20)), out[i]))
if __name__ == '__main__':
    unittest.main()