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
import oneflow.unittest
import oneflow as flow
import numpy as np


class ModuleTest(flow.nn.Module):
    def __init__(self, in_features, out_features, contiguous: bool):
        super().__init__()
        if contiguous:
            self.weight = flow.nn.Parameter(flow.ones(in_features, out_features))
        else:
            self.weight = flow.nn.Parameter(
                flow.ones(out_features, in_features).transpose(0, 1)
            )

    def forward(self, input):
        res = flow.matmul(input, self.weight)
        return res


def _test_graph_non_contiguous_tensors(test_case, device):
    class GraphTestContiguousTensors(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = ModuleTest(4, 3, True)

        def build(self, input):
            res = self.model(input)
            return res

    class GraphTestNonContiguousTensors(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.model = ModuleTest(4, 3, False)

        def build(self, input):
            res = self.model(input)
            return res

    graph_contiguous_tensors = GraphTestContiguousTensors()
    graph_non_contiguous_tensors = GraphTestNonContiguousTensors()

    test_case.assertTrue(graph_contiguous_tensors.model.weight.origin.is_contiguous())
    test_case.assertFalse(
        graph_non_contiguous_tensors.model.weight.origin.is_contiguous()
    )

    inp = flow.tensor([[1, 2, 3], [4, 5, 6], [3, 3, 3], [7, 8, 8]], dtype=flow.float32)

    non_contiguous_input = inp.transpose(0, 1)
    test_case.assertFalse(non_contiguous_input.is_contiguous())

    contiguous_input = non_contiguous_input.contiguous()
    test_case.assertTrue(contiguous_input.is_contiguous())

    contiguous_graph_output = graph_contiguous_tensors(contiguous_input)
    non_contiguous_graph_output = graph_non_contiguous_tensors(non_contiguous_input)

    test_case.assertTrue(
        np.array_equal(
            contiguous_graph_output.numpy(), non_contiguous_graph_output.numpy()
        )
    )


@flow.unittest.skip_unless_1n1d()
class TestGraphNonContiguousTensor(oneflow.unittest.TestCase):
    def test_graph_non_contiguous_tensors_gpu(test_case):
        _test_graph_non_contiguous_tensors(test_case)


if __name__ == "__main__":
    unittest.main()
