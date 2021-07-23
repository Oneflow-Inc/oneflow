import os
import random
import unittest

import numpy as np

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft


@flow.unittest.skip_unless_1n2d()
class TestShape(flow.unittest.TestCase):
    def test_shape(test_case):
        flow.clear_default_session()
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())

        @flow.global_function(function_config=func_config)
        def foo_job(input: oft.Numpy.Placeholder(shape=(2, 5))):
            ret = flow.identity(input)
            test_case.assertTrue(ret.shape == (1, 5))

        input_tensor = np.arange(10).reshape(2, 5).astype(np.single)
        foo_job(input_tensor)


if __name__ == "__main__":
    unittest.main()
