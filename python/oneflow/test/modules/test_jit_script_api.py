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
from collections import OrderedDict

import numpy as np

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.automated_test_util import *
from oneflow.test_utils.test_util import GenArgList


def _test_jit_script_api(test_case):
    @flow.jit.script
    def add2(x):
        return x + x

    x = flow.randn(2, 3)
    y = add2(x)
    test_case.assertTrue(x.size(), y.size())


def _test_jit_ignore_api(test_case):
    @flow.jit.ignore
    def add2(x):
        return x + x

    x = flow.randn(2, 3)
    y = add2(x)
    test_case.assertTrue(x.size(), y.size())


@flow.unittest.skip_unless_1n1d()
class TestJitScriptApi(flow.unittest.TestCase):
    def test_jit_script(test_case):
        arg_dict = OrderedDict()
        arg_dict["test_fun"] = [_test_jit_script_api, _test_jit_ignore_api]
        for arg in GenArgList(arg_dict):
            arg[0](test_case)


if __name__ == "__main__":
    unittest.main()
