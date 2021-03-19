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
import oneflow.typing as oft
from typing import Tuple
import time


@flow.unittest.skip_unless_1n4d()
class TestMultiProcess(flow.unittest.TestCase):
    def test_multi_process(test_case):
        flow.config.enable_debug_mode(True)
        flow.config.comm_net_worker_num(1)
        flow.config.gpu_device_num(4)
        func_config = flow.FunctionConfig()
        func_config.concurrency_width(1)

        @flow.global_function()
        def Foo():
            with flow.scope.placement("gpu", "0:0-3"):
                x = flow.get_variable(
                    "x",
                    shape=(2, 5),
                    dtype=flow.float,
                    initializer=flow.random_uniform_initializer(minval=0, maxval=1),
                    trainable=False,
                )
            return x

        of_ret = Foo().get()
        test_case.assertEqual(of_ret.numpy().shape, (2, 5))

    def test_worker_to_master_communication(test_case):
        flow.config.enable_debug_mode(True)
        flow.config.comm_net_worker_num(1)
        flow.config.gpu_device_num(4)
        func_config = flow.FunctionConfig()
        func_config.concurrency_width(1)

        @flow.global_function()
        def Foo():
            with flow.scope.placement("gpu", "0:0"):
                x = flow.get_variable(
                    "x",
                    shape=(2, 5),
                    dtype=flow.float,
                    initializer=flow.random_uniform_initializer(minval=0, maxval=1),
                    trainable=False,
                )
            with flow.scope.placement("gpu", "0:3"):
                y = flow.get_variable(
                    "y",
                    shape=(2, 5),
                    dtype=flow.float,
                    initializer=flow.constant_initializer(0),
                    trainable=False,
                )
                flow.assign(y, x)
            return y

        of_ret = Foo().get()
        test_case.assertEqual(of_ret.numpy().shape, (2, 5))

    def test_worker_to_worker_communication(test_case):
        flow.config.enable_debug_mode(True)
        flow.config.comm_net_worker_num(1)
        flow.config.gpu_device_num(4)
        func_config = flow.FunctionConfig()
        func_config.concurrency_width(1)

        @flow.global_function()
        def Foo():
            with flow.scope.placement("gpu", "0:1"):
                x = flow.get_variable(
                    "x",
                    shape=(2, 5),
                    dtype=flow.float,
                    initializer=flow.random_uniform_initializer(minval=0, maxval=1),
                    trainable=False,
                )
            with flow.scope.placement("gpu", "0:2"):
                y = flow.get_variable(
                    "y",
                    shape=(2, 5),
                    dtype=flow.float,
                    initializer=flow.constant_initializer(0),
                    trainable=False,
                )
                flow.assign(y, x)
            return y

        of_ret = Foo().get()
        test_case.assertEqual(of_ret.numpy().shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
