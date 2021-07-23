import unittest
import numpy as np
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft

def _test_tensor_buffer_convert(test_case):
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)
    func_config.default_logical_view(flow.scope.consistent_view())
    input_arr = np.random.rand(16, 24, 32, 36).astype(np.float32)

    @flow.global_function(function_config=func_config)
    def job_fn(x: oft.Numpy.Placeholder(input_arr.shape, dtype=flow.float32)):
        tensor_buffer = flow.tensor_to_tensor_buffer(x, instance_dims=2)
        return flow.tensor_buffer_to_tensor(tensor_buffer, dtype=flow.float32, instance_shape=[32, 36])
    output_arr = job_fn(input_arr).get().numpy()
    test_case.assertTrue(np.array_equal(input_arr, output_arr))

@flow.unittest.skip_unless_1n1d()
class TestTensorBufferOps(flow.unittest.TestCase):

    def test_tensor_buffer_convert(test_case):
        _test_tensor_buffer_convert(test_case)
if __name__ == '__main__':
    unittest.main()