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
from test_util import GenArgList

import oneflow as flow
import oneflow.unittest


@flow.unittest.skip_unless_1n1d()
# @unittest.skipIf(True, "skip until one_rec dataset is ready on CI server")
class TestOneRecOpsModule(flow.unittest.TestCase):
    def test_read_decode(test_case):
        files = [
            "/home/yaochi/dataset/onerec/part-00000-713a0aee-1337-4686-b418-0ada6face4de-c000.onerec"
        ]
        readdata = flow.read_onerec(
            files, batch_size=10, random_shuffle=True, shuffle_mode="batch"
        )
        labels = flow.decode_onerec(
            readdata, key="labels", dtype=flow.int32, shape=(1,)
        )
        dense_fields = flow.decode_onerec(
            readdata, key="dense_fields", dtype=flow.float, shape=(13,)
        )
        test_case.assertTrue(labels.shape == (10, 1))
        test_case.assertTrue(dense_fields.shape == (10, 1))


if __name__ == "__main__":
    unittest.main()
