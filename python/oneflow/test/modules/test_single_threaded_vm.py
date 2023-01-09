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


class TestSingleThreadedVM(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n2d()
    def test_ddp_in_single_threaded_vm(test_case):
        # Environment variables of current process like ONEFLOW_TEST_DEVICE_NUM
        # and environment variables about distributed training (i.e. MASTER_ADDR,
        # MASTER_PORT, WORLD_SIZE, RANK) are all in `env`.
        env = os.environ.copy()
        env["ONEFLOW_VM_MULTI_THREAD"] = "0"
        p = subprocess.run(
            [sys.executable, "test_ddp.py"],
            cwd=os.path.dirname(os.path.realpath(__file__)),
            env=env,
        )
        test_case.assertEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()
