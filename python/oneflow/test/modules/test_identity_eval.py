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
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
class TestIdentityEval(flow.unittest.TestCase):
    def test_simple(test_case):
        flow._C.identity_eval(flow.ones(1, 2, 3), 'print("TestIdentityEval.test_simple")')

    def test_fork_in_opkernel(test_case):
        code = "import os; os._exit(0) if os.fork() <= 0 else os.wait()"
        flow._C.identity_eval(flow.ones(1, 2, 3), 'exec("%s")' % code)
        flow._oneflow_internal.eager.Sync()


if __name__ == "__main__":
    unittest.main()
