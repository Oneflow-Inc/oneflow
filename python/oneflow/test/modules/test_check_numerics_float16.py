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

import os
import subprocess
import sys
import unittest

import oneflow as flow
import oneflow.unittest

_SCRIPT = """
import sys
import oneflow as flow

class G(flow.nn.Graph):
    def build(self, x):
        return x * 2

x = flow.tensor([1.0, 2.0, float(sys.argv[1])], device="cuda", dtype=flow.float16)
print("result", G()(x).tolist(), flush=True)
"""


def _run(value):
    env = dict(os.environ, ONEFLOW_DEBUG_KERNEL_SYNC_CHECK_NUMERICS="1")
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT, value],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        timeout=600,
    )


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestCheckNumericsFloat16(flow.unittest.TestCase):
    def test_finite(test_case):
        proc = _run("3.0")
        test_case.assertEqual(proc.returncode, 0, proc.stdout)
        test_case.assertIn("result [2.0, 4.0, 6.0]", proc.stdout)

    def test_not_finite(test_case):
        proc = _run("inf")
        test_case.assertNotEqual(proc.returncode, 0, proc.stdout)
        test_case.assertIn("has nan or inf", proc.stdout)


if __name__ == "__main__":
    unittest.main()
