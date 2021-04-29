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
import numpy as np
import oneflow as flow
from oneflow.python.nn.modules.dataset import OfrecordReader, raw_decoder


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestModule(flow.unittest.TestCase):
    def test_dataset(test_case):
        flow.InitEagerGlobalSession()

        @flow.global_function()
        def job():
            record_handle = OfrecordReader("/dataset/mnist_kaggle/6/train")()
            i = raw_decoder(record_handle, "img_raw", shape=(784,), dtype=flow.float32)
            test_case.assertTrue(type(record_handle) == flow.Tensor)
            test_case.assertTrue(type(i.numpy()) == np.ndarray)

        job()


if __name__ == "__main__":
    unittest.main()
