import numpy as np
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft
import unittest
import os

def TestReshape(x, shape, name):
    return flow.user_op_builder(name).Op('TestReshape').Input('in', [x]).Output('out').Attr('shape', shape).Build().InferAndTryRun().RemoteBlobList()[0]

def fixed_tensor_def_test(test_case, func_config):
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def ReshapeJob(x: oft.Numpy.Placeholder((10, 2))):
        return TestReshape(x, [5, 4], 'xx_test_reshape')
    x = np.random.rand(10, 2).astype(np.float32)
    y = ReshapeJob(x).get().numpy()
    print(y.shape)
    test_case.assertTrue((5, 4) == y.shape)
    test_case.assertTrue(np.array_equal(x.reshape(5, 4), y))

def mirrored_tensor_def_test(test_case, func_config):
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def ReshapeJob(x: oft.ListNumpy.Placeholder((10, 2))):
        return TestReshape(x, [5, 4], 'xx_test_reshape')
    x = np.random.rand(10, 2).astype(np.float32)
    y = ReshapeJob([x]).get().numpy_list()[0]
    test_case.assertTrue((5, 4) == y.shape)
    test_case.assertTrue(np.array_equal(x.reshape(5, 4), y))

@flow.unittest.skip_unless_1n1d()
class Test_TestReshape(flow.unittest.TestCase):

    @unittest.skipIf(os.getenv('ONEFLOW_TEST_CPU_ONLY'), 'only test cpu cases')
    def test_fixed_TestReshape(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())
        fixed_tensor_def_test(test_case, func_config)

    @unittest.skipIf(os.getenv('ONEFLOW_TEST_CPU_ONLY'), 'only test cpu cases')
    def test_mirrored_TestReshape(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())
        mirrored_tensor_def_test(test_case, func_config)
if __name__ == '__main__':
    unittest.main()