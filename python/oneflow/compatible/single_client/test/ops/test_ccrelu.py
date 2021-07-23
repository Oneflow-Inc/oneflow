import os
import unittest

import numpy as np

from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft


def ccrelu(x, name):
    return (
        flow.user_op_builder(name)
        .Op("ccrelu")
        .Input("in", [x])
        .Output("out")
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def fixed_tensor_def_test(test_case, func_config):
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def ReluJob(a: oft.Numpy.Placeholder((5, 2))):
        return ccrelu(a, "my_cc_relu_op")

    x = np.random.rand(5, 2).astype(np.float32)
    y = ReluJob(x).get().numpy()
    test_case.assertTrue(np.array_equal(y, np.maximum(x, 0)))


def mirrored_tensor_def_test(test_case, func_config):
    func_config.default_data_type(flow.float)

    @flow.global_function(function_config=func_config)
    def ReluJob(a: oft.ListNumpy.Placeholder((5, 2))):
        return ccrelu(a, "my_cc_relu_op")

    x = np.random.rand(3, 1).astype(np.float32)
    y = ReluJob([x]).get().numpy_list()[0]
    test_case.assertTrue(np.array_equal(y, np.maximum(x, 0)))


class TestCcrelu(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_ccrelu(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())
        fixed_tensor_def_test(test_case, func_config)

    @flow.unittest.skip_unless_1n1d()
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_mirror_ccrelu(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())
        mirrored_tensor_def_test(test_case, func_config)

    @flow.unittest.skip_unless_2n1d()
    def test_ccrelu_2n1c_0(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())
        fixed_tensor_def_test(test_case, func_config)

    @flow.unittest.skip_unless_2n1d()
    def test_ccrelu_2n1c_1(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())
        fixed_tensor_def_test(test_case, func_config)

    @flow.unittest.skip_unless_2n1d()
    def test_ccrelu_2n1c_2(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())
        fixed_tensor_def_test(test_case, func_config)

    @flow.unittest.skip_unless_2n1d()
    def test_ccrelu_2n1c_3(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.consistent_view())
        fixed_tensor_def_test(test_case, func_config)


if __name__ == "__main__":
    unittest.main()
