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
# RUN: python3 %s | FileCheck %s
from typing import Tuple
import unittest
import numpy as np
from numpy.core.fromnumeric import shape
import oneflow.compatible.single_client as flow
import oneflow.compatible.single_client.typing as oft
import oneflow.framework.dtype as dtype_util
from test_util import GenArgDict
from collections import OrderedDict


@flow.unittest.skip_unless_1n1d()
class TestMLIROptimizations(flow.unittest.TestCase):
    def test_cpu(self):
        d = OrderedDict(
            {"shape": [(96, 96), (3, 3)], "in_type": [flow.float32], "device": ["cpu"],}
        )
        for arg in GenArgDict(d):
            self.run_job(**arg)

    def test_gpu(self):
        d = OrderedDict(
            {"shape": [(96, 96), (3, 3)], "in_type": [flow.float32], "device": ["gpu"],}
        )
        for arg in GenArgDict(d):
            self.run_job(**arg)

    def run_job(test_case, device=None, in_type=None, shape=None):
        assert shape is not None
        flow.clear_default_session()
        func_config = flow.FunctionConfig()

        @flow.global_function(function_config=func_config)
        def FuseTrilScaleJob(
            x: oft.Numpy.Placeholder(shape, dtype=in_type)
        ) -> oft.Numpy:
            with flow.scope.placement(device, "0:0-0"):
                scale = 3.0
                loss = flow.math.tril(x * scale)
                loss += flow.math.tril(x) * scale
                return loss

        np_in_type = dtype_util.convert_oneflow_dtype_to_numpy_dtype(in_type)
        x = (np.random.rand(*shape) * 10).astype(np_in_type)
        FuseTrilScaleJob(x)


# cpu
# CHECK-LABEL: oneflow.job
# CHECK-LABEL: oneflow.job
# CHECK-LABEL: oneflow.job
# CHECK-LABEL: oneflow.job

# gpu
# CHECK-LABEL: oneflow.job
# CHECK: %0 = "oneflow.fused_scale_tril"
# CHECK: %1 = "oneflow.fused_scale_tril"
# CHECK-LABEL: oneflow.job
# CHECK: %0 = "oneflow.fused_scale_tril"
# CHECK: %1 = "oneflow.fused_scale_tril"

# CHECK-LABEL: oneflow.job
# CHECK: %0 = "oneflow.fused_scale_tril"
# CHECK: %1 = "oneflow.fused_scale_tril"
# CHECK-LABEL: oneflow.job
# CHECK: %0 = "oneflow.fused_scale_tril"
# CHECK: %1 = "oneflow.fused_scale_tril"

if __name__ == "__main__":
    unittest.main()
