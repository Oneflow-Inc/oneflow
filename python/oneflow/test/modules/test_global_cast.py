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
from re import X
import unittest

import oneflow as flow
from oneflow import nn
import oneflow.unittest
from oneflow.nn.graph.util import IONode


class GraphModel(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def build(self, x):
        return self.model(x)


class TestToGlobalLocal(oneflow.unittest.TestCase):
    placement = flow.placement('cpu', ranks=[0, 1])
    sbp = flow.sbp.broadcast
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    local_graph_model = GraphModel(model)
    global_graph_model = GraphModel(model.to_global(placement=placement, sbp=sbp))

    def test_none(test_case):
        x = None
        test_case.assertTrue(flow.to_global(x) is None)
        test_case.assertTrue(flow.to_local(x) is None)

    def test_tensor_to_global(test_case):
        local_tensor = flow.ones((3, 4))

        # local tensor -> global tensor
        global_tensor = flow.to_global(local_tensor, placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp)
        test_case.assertTrue(global_tensor.is_global)

        # global tensor -> global tensor
        global_tensor = flow.to_global(global_tensor, placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp)
        test_case.assertTrue(global_tensor.is_global)

        # wrong sbp type
        with test_case.assertRaises(TypeError):
            global_tensor = flow.to_global(local_tensor, placement=TestToGlobalLocal.placement, sbp=(TestToGlobalLocal.sbp, 0))

    def test_tensor_to_local(test_case):
        # global tensor -> local tensor
        global_tensor = flow.ones((3, 4), placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp)
        local_tensor = flow.to_local(global_tensor)
        test_case.assertFalse(local_tensor.is_global)

    def _test_state_dict_to_global(test_case, local_state_dict):
        # local state dict -> global state dict
        global_state_dict = flow.to_global(local_state_dict,
                                           placement=TestToGlobalLocal.placement,
                                           sbp=TestToGlobalLocal.sbp)
        node_tree = IONode(global_state_dict)
        for _, node in node_tree.named_nodes():
            if isinstance(node, flow.Tensor):
                test_case.assertTrue(node.is_global)

        # global state dict -> global state dict
        global_state_dict = flow.to_global(global_state_dict,
                                           placement=TestToGlobalLocal.placement,
                                           sbp=TestToGlobalLocal.sbp)
        node_tree = IONode(global_state_dict)
        for _, node in node_tree.named_nodes():
            if isinstance(node, flow.Tensor):
                test_case.assertTrue(node.is_global)

    def _test_state_dict_to_local(test_case, global_state_dict):
        # global state dict -> local state dict
        local_state_dict = flow.to_local(global_state_dict)
        node_tree = IONode(local_state_dict)
        for _, node in node_tree.named_nodes():
            if isinstance(node, flow.Tensor):
                test_case.assertFalse(node.is_global)

    def test_eagar_state_dict(test_case):
        test_case._test_state_dict_to_global(TestToGlobalLocal.model.state_dict())
        global_model = TestToGlobalLocal.model.to_global(placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp)
        test_case._test_state_dict_to_local(global_model.state_dict())

    def test_graph_state_dict(test_case):
        test_case._test_state_dict_to_global(TestToGlobalLocal.local_graph_model.state_dict())
        test_case._test_state_dict_to_local(TestToGlobalLocal.global_graph_model.state_dict())


if __name__ == "__main__":
    unittest.main()
