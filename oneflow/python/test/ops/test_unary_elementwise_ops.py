"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import unittest
import numpy as np
import oneflow as flow
from scipy.special import erf, erfc, gammaln
import oneflow.typing as oft
import os


@flow.unittest.skip_unless_1n2d()
class TestUnaryElementwiseOps(flow.unittest.TestCase):
    def test_abs(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def AbsJob(a: oft.Numpy.Placeholder((5, 2))):
            return flow.math.abs(a)

        x = np.random.rand(5, 2).astype(np.float32)
        y = AbsJob(x).get().numpy()
        test_case.assertTrue(np.array_equal(y, np.absolute(x)))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_1n2c_mirror_dynamic_abs(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())
        func_config.default_data_type(flow.float)

        @flow.global_function(function_config=func_config)
        def AbsJob(a: oft.ListNumpy.Placeholder((5, 2))):
            return flow.math.abs(a)

        x1 = np.random.rand(3, 1).astype(np.float32)
        x2 = np.random.rand(4, 2).astype(np.float32)
        y1, y2 = AbsJob([x1, x2]).get().numpy_list()
        test_case.assertTrue(np.array_equal(y1, np.absolute(x1)))
        test_case.assertTrue(np.array_equal(y2, np.absolute(x2)))

    def test_acos(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def AcosJob(a: oft.Numpy.Placeholder((5, 2))):
            return flow.math.acos(a)

        x = np.random.rand(5, 2).astype(np.float32)
        y = AcosJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.arccos(x)))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_acos_consistent_1n2c(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def AcosJob(a: oft.Numpy.Placeholder((5, 2))):
            return flow.math.acos(a)

        x = np.random.rand(5, 2).astype(np.float32)
        y = AcosJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.arccos(x)))

    def test_acos_cpu(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_placement_scope(flow.scope.placement("cpu", "0:0"))
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def AcosJob(a: oft.Numpy.Placeholder((5, 2))):
            return flow.math.acos(a)

        x = np.random.rand(5, 2).astype(np.float32)
        y = AcosJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.arccos(x)))

    def test_acos_double(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def AcosJob(a: oft.Numpy.Placeholder((5, 2), dtype=flow.double)):
            return flow.math.acos(a)

        x = np.random.rand(5, 2).astype(np.double)
        y = AcosJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.arccos(x)))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_1n2c_mirror_dynamic_acos(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_logical_view(flow.scope.mirrored_view())
        func_config.default_data_type(flow.float)

        @flow.global_function(function_config=func_config)
        def AcosJob(a: oft.ListNumpy.Placeholder((5, 2))):
            return flow.math.acos(a)

        x1 = np.random.rand(3, 1).astype(np.float32)
        x2 = np.random.rand(4, 2).astype(np.float32)
        y1, y2 = AcosJob([x1, x2]).get().numpy_list()
        test_case.assertTrue(np.allclose(y1, np.arccos(x1)))
        test_case.assertTrue(np.allclose(y2, np.arccos(x2)))

    def test_acosh(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def AcoshJob(a: oft.Numpy.Placeholder((7,))):
            return flow.math.acosh(a)

        # x = np.random.rand(7,).astype(np.float32)
        x = np.array([-2, -0.5, 1, 1.2, 200, 10000, float("inf")], dtype=np.float32)
        y = AcoshJob(x).get().numpy()
        # input: [-2, -0.5, 1, 1.2, 200, 10000, float("inf")]
        # output: [nan nan 0. 0.62236255 5.9914584 9.903487 inf]
        test_case.assertTrue(np.allclose(y, np.arccosh(x), equal_nan=True))

        x = np.random.uniform(low=1.0, high=100.0, size=(7,)).astype(np.float32)
        y = AcoshJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.arccosh(x), equal_nan=True))

    def test_asin(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def AsinJob(a: oft.Numpy.Placeholder((2,))):
            return flow.math.asin(a)

        x = np.array([0.8659266, 0.7068252], dtype=np.float32)
        y = AsinJob(x).get().numpy()
        # output: [1.047, 0.785] ~= [(PI/3), (PI/4)]
        test_case.assertTrue(np.allclose(y, np.arcsin(x), equal_nan=True))

        x = np.random.uniform(low=-1.0, high=1.0, size=(2,)).astype(np.float32)
        y = AsinJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.arcsin(x), equal_nan=True))

    def test_asinh(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def AsinhJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.asinh(a)

        x = np.array(
            [-float("inf"), -2, -0.5, 1, 1.2, 200, 10000, float("inf")],
            dtype=np.float32,
        )
        y = AsinhJob(x).get().numpy()
        # output: [-inf -1.4436355 -0.4812118 0.8813736 1.0159732 5.991471 9.903487 inf]
        test_case.assertTrue(np.allclose(y, np.arcsinh(x), equal_nan=True))
        # print("asinh y = ", y)

        x = np.random.uniform(size=(8,)).astype(np.float32)
        y = AsinhJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.arcsinh(x), equal_nan=True))

    def test_atan(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def AtanJob(a: oft.Numpy.Placeholder((2,))):
            return flow.math.atan(a)

        x = np.array([1.731261, 0.99920404], dtype=np.float32)
        y = AtanJob(x).get().numpy()
        # output: [1.047, 0.785] ~= [(PI/3), (PI/4)]
        test_case.assertTrue(np.allclose(y, np.arctan(x), equal_nan=True))
        # print("atan y = ", y)

        pi = 3.14159265357
        x = np.random.uniform(low=-pi / 2, high=pi / 2, size=(2,)).astype(np.float32)
        y = AtanJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.arctan(x), equal_nan=True))

    def test_atanh(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def AtanhJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.atanh(a)

        x = np.array(
            [-float("inf"), -1, -0.5, 1, 0, 0.5, 10, float("inf")], dtype=np.float32
        )
        y = AtanhJob(x).get().numpy()
        # output: [nan -inf -0.54930615 inf  0. 0.54930615 nan nan]
        test_case.assertTrue(np.allclose(y, np.arctanh(x), equal_nan=True))
        # print("atanh y = ", y)

        x = np.random.uniform(size=(8,)).astype(np.float32)
        y = AtanhJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.arctanh(x), equal_nan=True))

    def test_ceil(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def CeilJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.ceil(a)

        x = np.random.uniform(low=-10.0, high=10.0, size=(8,)).astype(np.float32)
        y = CeilJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.ceil(x), equal_nan=True))

    def test_cos(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def CosJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.cos(a)

        x = np.array(
            [-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")],
            dtype=np.float32,
        )
        y = CosJob(x).get().numpy()
        # output: [nan -0.91113025 0.87758255 0.5403023 0.36235774 0.48718765 -0.95215535 nan]
        test_case.assertTrue(np.allclose(y, np.cos(x), equal_nan=True))
        # print("cos y = ", y)

        x = np.random.uniform(size=(8,)).astype(np.float32)
        y = CosJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.cos(x), equal_nan=True))

    def test_cosh(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def CoshJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.cosh(a)

        x = np.array(
            [-float("inf"), -9, -0.5, 1, 1.2, 2, 10, float("inf")], dtype=np.float32
        )
        y = CoshJob(x).get().numpy()
        # output: [inf 4.0515420e+03 1.1276259e+00 1.5430807e+00 1.8106556e+00 3.7621956e+00 1.1013233e+04 inf]
        test_case.assertTrue(np.allclose(y, np.cosh(x), equal_nan=True))
        # print("cosh y = ", y)

        x = np.random.uniform(size=(8,)).astype(np.float32)
        y = CoshJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.cosh(x), equal_nan=True))

    def test_erf(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def ErfJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.erf(a)

        x = np.random.uniform(size=(8,)).astype(np.float32)
        y = ErfJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, erf(x), equal_nan=True))

    def test_erfc(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def ErfcJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.erfc(a)

        x = np.random.uniform(size=(8,)).astype(np.float32)
        y = ErfcJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, erfc(x), equal_nan=True))

    def test_exp(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def ExpJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.exp(a)

        x = np.random.uniform(size=(8,)).astype(np.float32)
        y = ExpJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.exp(x), equal_nan=True))

    def test_expm1(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Expm1Job(a: oft.Numpy.Placeholder((8,))):
            return flow.math.expm1(a)

        x = np.random.uniform(size=(8,)).astype(np.float32)
        y = Expm1Job(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.expm1(x), equal_nan=True))

    def test_floor(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def FloorJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.floor(a)

        x = np.random.uniform(low=-10.0, high=10.0, size=(8,)).astype(np.float32)
        y = FloorJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.floor(x), equal_nan=True))

    def test_lgamma(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def LgammaJob(a: oft.Numpy.Placeholder((6,))):
            return flow.math.lgamma(a)

        x = np.array([0, 0.5, 1, 4.5, -4, -5.6], dtype=np.float32)
        y = LgammaJob(x).get().numpy()
        # output: [inf, 0.5723649, 0., 2.4537368, inf, -4.6477685]
        # print("lgamma y = ", y)
        test_case.assertTrue(np.allclose(y, gammaln(x), equal_nan=True))

    def test_log(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def LogJob(a: oft.Numpy.Placeholder((4,))):
            return flow.math.log(a)

        x = np.array([0, 0.5, 1, 5], dtype=np.float32)
        y = LogJob(x).get().numpy()
        # output: [-inf, -0.6931472,  0. ,  1.609438]
        # print("log y = ", y)
        test_case.assertTrue(np.allclose(y, np.log(x), equal_nan=True))

    def test_log1p(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def Log1pJob(a: oft.Numpy.Placeholder((4,))):
            return flow.math.log1p(a)

        x = np.array([0, 0.5, 1, 5], dtype=np.float32)
        y = Log1pJob(x).get().numpy()
        # output: [0., 0.4054651, 0.6931472, 1.791759]
        # print("log1p y = ", y)
        test_case.assertTrue(np.allclose(y, np.log1p(x), equal_nan=True))

    def test_log_sigmoid(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def LogSigmoidJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.log_sigmoid(a)

        x = np.random.uniform(low=-5.0, high=5.0, size=(8,)).astype(np.float32)
        y = LogSigmoidJob(x).get().numpy()
        # print("log_sigmoid y = ", y)
        test_case.assertTrue(
            np.allclose(
                y, -np.log(1 + np.exp(-x)), equal_nan=True, rtol=1e-03, atol=1e-05
            )
        )

    def test_negative(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def NegativeJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.negative(a)

        x = np.random.uniform(low=-10.0, high=10.0, size=(8,)).astype(np.float32)
        y = NegativeJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, -x, equal_nan=True))

    def test_reciprocal(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def ReciprocalJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.reciprocal(a)

        x = np.random.uniform(low=-10.0, high=10.0, size=(8,)).astype(np.float32)
        y = ReciprocalJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, 1.0 / x, equal_nan=True))

    def test_reciprocal_no_nan(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def ReciprocalNoNanJob(a: oft.Numpy.Placeholder((4,))):
            return flow.math.reciprocal_no_nan(a)

        x = np.array([2.0, 0.5, 0, 1], dtype=np.float32)
        out = np.array([0.5, 2, 0.0, 1.0], dtype=np.float32)
        y = ReciprocalNoNanJob(x).get().numpy()
        # print("reciprocal_no_nan: y = ", y)
        test_case.assertTrue(np.allclose(y, out, equal_nan=True))

    def test_rint(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def RintJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.rint(a)

        x = np.random.uniform(low=-10.0, high=10.0, size=(8,)).astype(np.float32)
        y = RintJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.rint(x), equal_nan=True))

    def test_rint_special_value(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def RintJob(a: oft.Numpy.Placeholder((9,))):
            return flow.math.rint(a)

        x = np.array(
            [0.5000001, -1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.5, 3.5], dtype=np.float32
        )
        out = np.array(
            [1.0, -2.0, -2.0, -0.0, 0.0, 2.0, 2.0, 2.0, 4.0], dtype=np.float32
        )
        y = RintJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, out, equal_nan=True))

    def test_round(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def RoundJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.round(a)

        x = np.random.uniform(low=-10.0, high=10.0, size=(8,)).astype(np.float32)
        y = RoundJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.round(x), equal_nan=True))

    def test_round_special_value(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def RoundJob(a: oft.Numpy.Placeholder((5,))):
            return flow.math.round(a)

        x = np.array([0.9, 2.5, 2.3, 1.5, -4.5], dtype=np.float32)
        out = np.array([1.0, 2.0, 2.0, 2.0, -4.0], dtype=np.float32)
        y = RoundJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, out, equal_nan=True))

    def test_rsqrt(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def RsqrtJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.rsqrt(a)

        x = np.random.uniform(low=-10.0, high=10.0, size=(8,)).astype(np.float32)
        y = RsqrtJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, 1 / np.sqrt(x), equal_nan=True))

    def test_sigmoid_v2(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def SigmoidJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.sigmoid_v2(a)

        x = np.random.uniform(low=-2.0, high=2.0, size=(8,)).astype(np.float32)
        y = SigmoidJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, 1.0 / (1.0 + np.exp(-x)), equal_nan=True))

    def test_sign(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def SignJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.sign(a)

        x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.float32)
        y = SignJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.sign(x), equal_nan=True))

    def test_sign_double(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def SignJob(a: oft.Numpy.Placeholder((8,), dtype=flow.double)):
            return flow.math.sign(a)

        x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.double)
        y = SignJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.sign(x), equal_nan=True))

    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_sign_double_consistent_1n2c(test_case):
        flow.config.gpu_device_num(2)
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def SignJob(a: oft.Numpy.Placeholder((8,), dtype=flow.double)):
            return flow.math.sign(a)

        x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.double)
        y = SignJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.sign(x), equal_nan=True))

    def test_sin(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def SinJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.sin(a)

        x = np.array(
            [-float("inf"), -9, -0.5, 1, 1.2, 200, 10, float("inf")], dtype=np.float32
        )
        y = SinJob(x).get().numpy()
        # output: [nan -0.4121185 -0.47942555 0.84147096 0.9320391 -0.87329733 -0.54402107 nan]
        test_case.assertTrue(np.allclose(y, np.sin(x), equal_nan=True))

        x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.float32)
        y = SinJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.sin(x), equal_nan=True))

    def test_softplus(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def SoftplusJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.softplus(a)

        x = np.random.uniform(low=-10.0, high=10.0, size=(8,)).astype(np.float32)
        y = SoftplusJob(x).get().numpy()
        test_case.assertTrue(
            np.allclose(
                y, np.log(np.exp(x) + 1), equal_nan=True, rtol=1e-03, atol=1e-05
            )
        )

    def test_sqrt(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def SqrtJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.sqrt(a)

        x = np.random.uniform(low=0.0, high=100.0, size=(8,)).astype(np.float32)
        y = SqrtJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.sqrt(x), equal_nan=True))

    def test_square(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def SquareJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.square(a)

        x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.float32)
        y = SquareJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, x * x, equal_nan=True))

    def test_tan(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def TanJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.tan(a)

        x = np.array(
            [-float("inf"), -9, -0.5, 1, 1.2, 200, 10000, float("inf")],
            dtype=np.float32,
        )
        y = TanJob(x).get().numpy()
        # output: [nan 0.45231566 -0.5463025 1.5574077 2.572152 -1.7925274 0.32097113 nan]
        test_case.assertTrue(np.allclose(y, np.tan(x), equal_nan=True))

        x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.float32)
        y = TanJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.tan(x), equal_nan=True))

    def test_tanh(test_case):
        func_config = flow.FunctionConfig()
        func_config.default_data_type(flow.float)
        func_config.default_logical_view(flow.scope.consistent_view())

        @flow.global_function(function_config=func_config)
        def TanhJob(a: oft.Numpy.Placeholder((8,))):
            return flow.math.tanh(a)

        x = np.array(
            [-float("inf"), -5, -0.5, 1, 1.2, 2, 3, float("inf")], dtype=np.float32
        )
        y = TanhJob(x).get().numpy()
        # output: [-1. -0.99990916 -0.46211717 0.7615942 0.8336547 0.9640276 0.9950547 1.]
        test_case.assertTrue(np.allclose(y, np.tanh(x), equal_nan=True))

        x = np.random.uniform(low=-100.0, high=100.0, size=(8,)).astype(np.float32)
        y = TanhJob(x).get().numpy()
        test_case.assertTrue(np.allclose(y, np.tanh(x), equal_nan=True))


if __name__ == "__main__":
    unittest.main()
