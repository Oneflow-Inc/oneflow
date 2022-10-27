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
import oneflow as flow
import oneflow.unittest
import unittest


@flow.unittest.skip_unless_1n1d()
class TestTestForkOnWorkerThread(flow.unittest.TestCase):
    def test_test_fork_on_worker_thread(test_case):
        flow._oneflow_internal._C.test_fork_on_worker_thread(flow.ones(32))


if __name__ == "__main__":
    unittest.main()
