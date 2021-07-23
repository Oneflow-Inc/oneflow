import numpy as np
from oneflow.compatible import single_client as flow
import unittest
import os


@flow.unittest.skip_unless_1n2d()
class TestDistributeConcat(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_deadlock(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.enable_inplace(False)

        @flow.global_function(function_config=func_config)
        def DistributeConcat():
            with flow.scope.placement("gpu", "0:0"):
                w = flow.get_variable(
                    "w", (2, 5), initializer=flow.constant_initializer(10)
                )
                x = w + 1
                y = w + 1
            ret = flow.advanced.distribute_concat([x, y])

        DistributeConcat()


if __name__ == "__main__":
    unittest.main()
