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

import oneflow as flow
import oneflow.unittest


class MyModule(flow.nn.Module):
    def __init__(self, placement=None, sbp=None):
        super().__init__()
        w = flow.randn(10, 10, placement=placement, sbp=sbp)
        self.weight = flow.nn.Parameter(w)

    def forward(self, input):
        return flow._C.gather(self.weight, input, 0)


class MyGraph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module
        sgd = flow.optim.SGD(module.parameters(), lr=1e-3)
        self.add_optimizer(sgd, is_sparse=True)

    def build(self, input):
        result = self.m(input)
        result.mean().backward()


def _rand_input(placement=None, sbp=None):
    generator = flow.Generator()
    generator.manual_seed(0)
    return flow.randint(0, 10, (8,), generator=generator, placement=placement, sbp=sbp)


@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
@flow.unittest.skip_unless_1n1d()
class GraphSparseOptimizerTest(oneflow.unittest.TestCase):
    def test(test_case):
        PLC = flow.placement("cuda", ranks=[0])
        SBP = flow.sbp.broadcast
        m = MyModule(PLC, SBP)
        graph = MyGraph(m)
        graph._compile(_rand_input(PLC, SBP))

        sparse_optimizer_found = False
        for op in graph._full_graph_proto.net.op:
            # print("==>", op.name)
            if op.HasField("user_conf"):
                # print("  -->", op.user_conf.op_type_name)
                if op.user_conf.op_type_name == "indexed_slices_sgd_update":
                    sparse_optimizer_found = True
                    break

        test_case.assertTrue(sparse_optimizer_found)


if __name__ == "__main__":
    unittest.main()
