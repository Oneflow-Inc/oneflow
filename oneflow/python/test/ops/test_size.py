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
import numpy as np
import os
import random

@flow.unittest.skip_unless_1n1d()
class TestSize(flow.unittest.TestCase):
    def test_size(test_case):
        size = flow.Size((4, 3, 10, 5))
        test_case.assertTrue(size[0] == 4)
        test_case.assertTrue(size[2] == 10)
        size[1] = 10
        test_case.assertTrue(size[1] == 10)
        test_case.assertTrue(len(size) == 4)

    def test_unpack(test_case):
        one, two, three, four = flow.Size((1, 2, 3, 4))
        test_case.assertEqual(one, 1)
        test_case.assertEqual(two, 2)
        test_case.assertEqual(three, 3)
        test_case.assertEqual(four, 4)

if __name__ == "__main__":
    unittest.main()
