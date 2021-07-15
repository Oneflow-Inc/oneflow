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
import oneflow as flow
from test_util import GenArgList
import unittest
import os


def do_test(test_case, mirrored):
    flow.clear_default_session()
    flow.config.gpu_device_num(2)
    func_config = flow.FunctionConfig()

    if mirrored:
        func_config.default_logical_view(flow.scope.mirrored_view())
    else:
        func_config.default_logical_view(flow.scope.consistent_view())

    @flow.global_function(type="train", function_config=func_config)
    def Foo():
        w = flow.get_variable("w", (10,), initializer=flow.constant_initializer(1))
        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [5])
        flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(w)
        return w

    r1 = Foo().get().numpy()
    test_case.assertTrue(np.all(r1 == 1.0))
    r2 = Foo().get().numpy()
    test_case.assertTrue(np.all(r2 == 0.5))


@flow.unittest.skip_unless_1n2d()
class TestAllReduceGroup(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_variable_as_loss_on_two_device(test_case):
        arg_dict = OrderedDict()
        arg_dict["mirrored"] = [True, False]
        for arg in GenArgList(arg_dict):
            do_test(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
