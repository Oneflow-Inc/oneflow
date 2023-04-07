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
import subprocess
import sys
import os
import unittest
import oneflow as flow
import oneflow.unittest


class TestRemat(flow.unittest.TestCase):
    def test_remat_in_single_threaded_vm(test_case):
        env = os.environ.copy()
        env["ONEFLOW_VM_MULTI_THREAD"] = "0"
        p = subprocess.run(
            [sys.executable, "_test_remat.py"],
            cwd=os.path.dirname(os.path.realpath(__file__)),
            env=env,
        )
        test_case.assertEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()
