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


def _get_regularizer(model_name):
    # all decay
    return flow.regularizers.l2(0.00004)


def _batch_norm(inputs, last=False):
    initializer = flow.zeros_initializer() if last else flow.ones_initializer()
    axis = 1
    weight_regularizer = flow.regularizers.l2(0.5)
    trainable = True
    training = True
    data_format = "NHWC"
    if data_format == "NHWC":
        axis = 3
    return flow.layers.batch_normalization(
        inputs=inputs,
        axis=axis,
        momentum=0.9,  # 97,
        epsilon=1e-5,
        center=True,
        scale=True,
        trainable=trainable,
        training=training,
        gamma_initializer=initializer,
        moving_variance_initializer=initializer,
        gamma_regularizer=weight_regularizer,
        beta_regularizer=weight_regularizer,
    )


@flow.unittest.skip_unless_1n1d()
class TestMLIROptimizations(flow.unittest.TestCase):
    @unittest.skip("")
    def test_cpu(self):
        d = OrderedDict(
            {"shape": [(2, 96, 96, 3)], "in_type": [flow.float32], "device": ["cpu"],}
        )
        for arg in GenArgDict(d):
            self.run_job(**arg)

    def test_gpu(self):
        d = OrderedDict(
            {"shape": [(2, 96, 96, 3)], "in_type": [flow.float32], "device": ["gpu"],}
        )
        for arg in GenArgDict(d):
            self.run_job(**arg)

    def run_job(test_case, device=None, in_type=None, shape=None):
        assert shape is not None
        flow.clear_default_session()
        func_config = flow.FunctionConfig()

        @flow.global_function(type="train", function_config=func_config)
        def FuseBnAddReluJob(
            x: oft.Numpy.Placeholder(shape, dtype=in_type)
        ) -> oft.Numpy:
            addend = flow.constant_like(x, 2)
            with flow.scope.placement(device, "0:0-0"):
                x = (
                    flow.get_variable(
                        "x1",
                        shape=shape,
                        dtype=in_type,
                        initializer=flow.random_uniform_initializer(
                            minval=-10, maxval=10
                        ),
                        trainable=True,
                    )
                    + x
                )
                loss = flow.nn.relu(_batch_norm(x, last=False) + addend) + 1
                flow.optimizer.SGD(
                    flow.optimizer.PiecewiseConstantScheduler([], [0.0001]), momentum=0
                ).minimize(loss)
                return loss

        np_in_type = dtype_util.convert_oneflow_dtype_to_numpy_dtype(in_type)
        x = (np.random.rand(*shape) * 10).astype(np_in_type)
        FuseBnAddReluJob(x)


# CHECK: [[RESULT_1:%y.*]], [[RESULT_2:%reserve_space.*]], [[RESULT_3:%mean.*]], [[RESULT_4:%inv_variance.*]] = "oneflow.normalization_add_relu"

if __name__ == "__main__":
    unittest.main()
