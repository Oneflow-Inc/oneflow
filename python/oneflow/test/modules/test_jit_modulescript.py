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

from oneflow.test_utils.automated_test_util import *
import oneflow as flow

import numpy as np

from oneflow.test_utils.test_util import GenArgList

@flow.unittest.skip_unless_1n1d()
class test_jit_scriptmodule(flow.unittest.TestCase):
    def testcase4module(test_case):
        model = flow.nn.Sequential(
            flow.nn.Linear(5, 3),
            flow.nn.Linear(3, 1)
            )
        status = isinstance(model, flow.jit.ScriptModule)
        test_case.assertFalse(status, False)

if __name__ == "__main__":
    unittest.main()
    
