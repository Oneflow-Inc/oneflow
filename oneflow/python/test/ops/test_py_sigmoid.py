
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
from collections import OrderedDict

import numpy as np
import math
import oneflow as flow
import oneflow.typing as oft
from test_util import Args, CompareOpWithTensorFlow, GenArgDict

flow.config.enable_debug_mode(True)

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def numpy_sigmoid(x):
    return 1/(1 + np.exp(-x))


def make_job(input_shape, dtype=flow.float32):
    @flow.global_function(function_config=func_config)
    def sigmoid_job(x: oft.Numpy.Placeholder(input_shape, dtype=dtype)):
        return flow.math.sigmoid(x)

    return sigmoid_job


def make_py_job(input_shape, dtype=flow.float32):
    @flow.global_function(function_config=func_config)
    def py_sigmoid_job(x: oft.Numpy.Placeholder(input_shape, dtype=dtype)):
        with flow.scope.placement("cpu", "0:0"):
            return flow.math.sigmoid_py(x)

    return py_sigmoid_job


def test_py_sigmoid(test_case):
    x = np.ones((1, 10), dtype=np.float32)

    sig_job = make_job(x.shape)
    py_sig_job = make_py_job(x.shape)

    sig = sig_job(x).get().numpy()
    py_sig = py_sig_job(x).get().numpy()
    numpy_sig = numpy_sigmoid(x)
    print("sig : ", sig)
    print("py_sig : ", py_sig)
    print("numpy_sig : ", numpy_sig)

    test_case.assertTrue(np.allclose(
        sig, py_sig, rtol=1e-03, atol=1e-05))
    test_case.assertTrue(np.allclose(
        py_sig, numpy_sig, rtol=1e-03, atol=1e-05))
