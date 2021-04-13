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
import oneflow.typing as tp


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in eager mode",
)
class TestModule(flow.unittest.TestCase):
    def test_dataset(test_case):
        flow.InitEagerGlobalSession()

        @flow.global_function()
        def job():
            record_handle = flow.tmp.OfrecordReader(
                "/dataset/lenet_mnist/data/ofrecord/train"
            )
            i = flow.tmp.RawDecoder(
                record_handle, "image_raw", shape=(784,), dtype=flow.float32
            )
            assert type(record_handle) == flow.Tensor
            assert type(i.numpy()) == np.ndarray

        job()

if __name__ == "__main__":
    unittest.main()
