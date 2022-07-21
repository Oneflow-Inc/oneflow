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
import oneflow.unittest


class DataLoaderGraph1(flow.nn.Graph):
    def __init__(self, loader):
        super().__init__()
        self.loader_ = loader

    def build(self):
        return self.loader_()


class DataLoaderGraph2(flow.nn.Graph):
    def __init__(self, loader0, loader1):
        super().__init__()
        self.loader0_ = loader0
        self.loader1_ = loader1

    def build(self):
        return self.loader0_(), self.loader1_()


class RawReaderTestCase(oneflow.unittest.TestCase):
    def test_case1(test_case):
        total_instances = 1024
        batch_size = 16
        instances_size = 128
        data = np.random.randn(total_instances, instances_size).astype(np.float32)
        with open("test.bin", "wb") as f:
            f.write(data.tobytes())
        raw_reader = flow.nn.RawReader(
            ["test.bin"],
            (instances_size,),
            flow.float32,
            batch_size,
            random_shuffle=False,
        )
        loader_graph = DataLoaderGraph1(raw_reader)
        outs = []
        for i in range(total_instances // batch_size):
            outs.append(loader_graph().numpy())
        out = np.concatenate(outs, axis=0)
        test_case.assertTrue(np.array_equal(out, data))
        outs = []
        for i in range(total_instances // batch_size):
            outs.append(loader_graph().numpy())
        out = np.concatenate(outs, axis=0)
        test_case.assertTrue(np.array_equal(out, data))

    def test_case2(test_case):
        total_instances = 1024
        batch_size = 16
        instances_size = 128
        data = np.random.randn(total_instances, instances_size).astype(np.float32)
        with open("test.bin", "wb") as f:
            f.write(data.tobytes())
        reader0 = flow.nn.RawReader(
            ["test.bin"],
            (instances_size,),
            flow.float32,
            batch_size,
            random_shuffle=True,
            random_seed=1234,
        )
        reader1 = flow.nn.RawReader(
            ["test.bin"],
            (instances_size,),
            flow.float32,
            batch_size,
            random_shuffle=True,
            random_seed=1234,
        )
        loader_graph = DataLoaderGraph2(reader0, reader1)
        for i in range(total_instances // batch_size * 2):
            out0, out1 = loader_graph()
            test_case.assertTrue(np.array_equal(out0.numpy(), out1.numpy()))


@flow.unittest.skip_unless_1n2d()
class RawReaderDistributedTestCase(flow.unittest.TestCase):
    def test_case1(test_case):
        total_instances = 1024
        batch_size = 16
        instances_size = 128
        np.random.seed(1234)
        data = np.random.randn(total_instances, instances_size).astype(np.float32)
        if flow.env.get_rank() == 0:
            with open("test.bin", "wb") as f:
                f.write(data.tobytes())
        flow._oneflow_internal.eager.Sync()
        raw_reader = flow.nn.RawReader(
            ["test.bin"],
            (instances_size,),
            flow.float32,
            batch_size,
            random_shuffle=False,
        )
        loader_graph = DataLoaderGraph1(raw_reader)
        outs = []
        for i in range(total_instances // batch_size):
            outs.append(loader_graph().numpy())
        out = np.concatenate(outs, axis=0)
        test_case.assertTrue(np.array_equal(out, data))
        outs = []
        for i in range(total_instances // batch_size):
            outs.append(loader_graph().numpy())
        out = np.concatenate(outs, axis=0)
        test_case.assertTrue(np.array_equal(out, data))


if __name__ == "__main__":
    unittest.main()
