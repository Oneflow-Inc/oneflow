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
import os
import shutil
import tempfile
import time
from collections import OrderedDict

from scipy.stats import ks_2samp
import numpy as np
import oneflow as flow
import oneflow.typing as tp
from test_util import GenArgDict


SHAPE = (4, 8, 5, 6)


def get_simple_model(dtype, initializer):
    @flow.global_function()
    def model() -> tp.Numpy:
        x = flow.get_variable(
            name="x", shape=SHAPE, dtype=dtype, initializer=initializer,
        )
        return x

    return model


def CompareTwoDistribution(test_case, dtype, initializer):
    flow.clear_default_session()
    flow.config.enable_legacy_model_io(True)
    model = get_simple_model(dtype, initializer)
    flow.train.CheckPoint().init()
    legacy_init_res = model()

    flow.clear_default_session()
    flow.config.enable_legacy_model_io(False)
    model = get_simple_model(dtype, initializer)
    new_init_res = model()

    s = ks_2samp(legacy_init_res.flatten(), new_init_res.flatten())
    pvalue = s.pvalue
    test_case.assertGreater(pvalue, 0.0001, msg=initializer)


class TestInitializer(flow.unittest.TestCase):
    def test_int_initializer(test_case):
        initializers = [
            flow.random_uniform_initializer(minval=-6, maxval=18, dtype=flow.int32),
            flow.constant_initializer(value=4, dtype=flow.int32),
        ]

        for initializer in initializers:
            CompareTwoDistribution(test_case, flow.int32, initializer)

    def test_float_initializer(test_case):
        initializers = [
            flow.random_normal_initializer(mean=3, stddev=4),
            flow.random_uniform_initializer(minval=-6, maxval=18),
            flow.truncated_normal_initializer(mean=-5, stddev=8),
            flow.xavier_uniform_initializer(data_format="NCHW"),
            flow.xavier_uniform_initializer(data_format="NHWC"),
            flow.xavier_normal_initializer(data_format="NCHW"),
            flow.xavier_normal_initializer(data_format="NHWC"),
            flow.constant_initializer(value=4),
            flow.ones_initializer(),
            flow.zeros_initializer(),
        ]

        kaiming_args = GenArgDict(
            OrderedDict(
                shape=[SHAPE],
                mode=["fan_in", "fan_out", "fan_avg"],
                distribution=["random_normal", "random_uniform"],
                data_format=["NCHW", "NHWC"],
                negative_slope=[0.5],
            )
        )
        vs_args = GenArgDict(
            OrderedDict(
                scale=[3.4],
                mode=["fan_in", "fan_out", "fan_avg"],
                distribution=["truncated_normal", "random_normal", "random_uniform"],
                data_format=["NCHW", "NHWC"],
            )
        )
        for args in kaiming_args:
            initializers.append(flow.kaiming_initializer(**args))

        for args in vs_args:
            initializers.append(flow.variance_scaling_initializer(**args))

        for initializer in initializers:
            CompareTwoDistribution(test_case, flow.float32, initializer)


if __name__ == "__main__":
    unittest.main()
