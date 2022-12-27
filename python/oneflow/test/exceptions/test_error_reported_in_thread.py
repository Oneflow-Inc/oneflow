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
import tempfile
import os
import unittest
import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
def test_error_reported_in_thread():
    for env_name in ["ONEFLOW_DEBUG", "ONEFLOW_PYTHON_STACK_GETTER"]:
        env = os.environ.copy()
        env[env_name] = "1"
        # Run a new process to capture the error output
        p = subprocess.run(
            [sys.executable, "throw_error.py"],
            capture_output=True,
            cwd=os.path.dirname(os.path.realpath(__file__)),
            env=env,
        )
        assert p.returncode != 0
        error_msg = p.stderr.decode("utf-8")
        print(error_msg)
        assert (
            """File "throw_error.py", line 19, in g
    flow._C.throw_error(x)
  File "throw_error.py", line 23, in f
    g(x)
  File "throw_error.py", line 26, in <module>
    f(x)"""
            in error_msg
        )


@flow.unittest.skip_unless_1n1d()
def test_python_stack_getter_disabled():
    # Run a new process to capture the error output
    p = subprocess.run(
        [sys.executable, "throw_error.py"],
        capture_output=True,
        cwd=os.path.dirname(os.path.realpath(__file__)),
    )
    assert p.returncode != 0
    error_msg = p.stderr.decode("utf-8")
    assert "No Python stack available." in error_msg
    assert "ONEFLOW_DEBUG" in error_msg
    assert "ONEFLOW_PYTHON_STACK_GETTER" in error_msg
