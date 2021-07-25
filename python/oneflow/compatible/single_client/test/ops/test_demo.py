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

import oneflow.compatible.single_client.unittest
from oneflow.compatible import single_client as flow


@unittest.skipIf(flow.unittest.env.device_num() != 1, "only runs when device_num is 1")
class TestDemo(flow.unittest.TestCase):
    @unittest.skipIf(
        flow.unittest.env.node_size() != 1, "only runs when node_size is 1"
    )
    def test_foo(test_case):
        pass

    @unittest.skipIf(
        flow.unittest.env.node_size() != 2, "only runs when node_size is 2"
    )
    def test_bar(test_case):
        pass


if __name__ == "__main__":
    unittest.main()
