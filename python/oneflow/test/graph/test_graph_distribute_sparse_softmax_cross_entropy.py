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
import unittest
import numpy as np

import oneflow as flow
import oneflow.unittest


def _test_distribute_softmax_entropy_graph(test_case, device):
    np_logits = np.random.randn(5, 10)
    np_labels = np.random.randint(0, 2, size=(5,))

    of_logits = flow.tensor(np_logits, device=device)
    of_labels = flow.tensor(np_labels, device=device, dtype=flow.int32)

    class DistributeSparseSoftmaxCrossEntropyMsGraph(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.op = flow.nn.functional.distributed_sparse_softmax_cross_entropy

        def build(self, logits, label):
            return self.op(logits, label)

    SSCE_g = DistributeSparseSoftmaxCrossEntropyMsGraph()
    SSCE_g.debug()
    of_lazy_out = SSCE_g(of_logits, of_labels)
    print(of_lazy_out.numpy())


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class TestLinearGraph(oneflow.unittest.TestCase):
    def test_linear_graph_gpu(test_case):
        _test_distribute_softmax_entropy_graph(test_case, flow.device("cuda"))

    def test_linear_graph_cpu(test_case):
        _test_distribute_softmax_entropy_graph(test_case, flow.device("cpu"))


if __name__ == "__main__":
    unittest.main()
