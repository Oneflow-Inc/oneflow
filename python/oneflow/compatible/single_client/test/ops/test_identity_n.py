import unittest
import numpy as np
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft
from typing import Tuple
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

@flow.unittest.skip_unless_1n1d()
class TestIdentityN(flow.unittest.TestCase):

    def test_identity_n(test_case):

        @flow.global_function(function_config=func_config)
        def identity_n_job(xs: Tuple[(oft.Numpy.Placeholder((5, 2)),) * 3]):
            return flow.identity_n(xs)
        inputs = tuple((np.random.rand(5, 2).astype(np.float32) for i in range(3)))
        res = identity_n_job(inputs).get()
        for i in range(3):
            test_case.assertTrue(np.array_equal(res[i].numpy(), inputs[i]))
if __name__ == '__main__':
    unittest.main()