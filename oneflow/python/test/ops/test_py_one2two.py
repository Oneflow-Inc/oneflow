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
import math
import numpy as np
import os
import unittest
from test_util import Args, CompareOpWithTensorFlow, GenArgDict

import oneflow as flow
import oneflow.typing as oft

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

lib_path = os.path.dirname(os.path.abspath(__file__))


@flow.unittest.skip_unless_1n1d()
class TestPyOne2Two(flow.unittest.TestCase):
    def test_py_one2two(test_case):
        py_one2two_lib = flow.util.op_lib("py_one2two", lib_path)
        py_one2two_lib.AddPythonAPI()
        py_one2two_lib.AddOpDef()
        py_one2two_lib.AddPythonKernel()
        py_one2two_lib.Build()

        op_lib_ld = flow.util.op_lib_loader()
        op_lib_ld.AddLib(py_one2two_lib)
        op_lib_ld.Link()
        op_lib_ld.Load()
        print(op_lib_ld.LibList())

        def make_py_job(input_shape, dtype=flow.float32):
            @flow.global_function(function_config=func_config)
            def py_job(x: oft.Numpy.Placeholder(input_shape, dtype=dtype)):
                with flow.scope.placement("cpu", "0:0"):
                    return py_one2two_lib.api.py_one2two(x)

            return py_job

        x = np.ones((1, 10), dtype=np.float32)
        py_job = make_py_job(x.shape)
        outs = py_job(x).get()
        for out in outs:
            print("out:", out.numpy())
            test_case.assertTrue(np.allclose(x, out.numpy(), rtol=1e-03, atol=1e-05))


if __name__ == "__main__":
    unittest.main()
