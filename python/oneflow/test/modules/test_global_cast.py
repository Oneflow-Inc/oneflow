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
from oneflow import nn
import oneflow.unittest
from oneflow.nn.graph.util import IONodeType, IONode


class GraphModel(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def build(self, x):
        return self.model(x)


class TestToGlobalLocal(oneflow.unittest.TestCase):
    placement = flow.placement('cpu', ranks=[0, 1])
    sbp = None
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    local_graph_model = GraphModel(model)
    global_graph_model = None

    def _all_global(test_case, input, placement, sbp):
        node_tree = IONode(value=input)
        for _, node in node_tree.named_nodes():
            if node._type == IONodeType.TENSOR:
                value = node._value
                test_case.assertTrue(value.is_global)
                # check placement
                test_case.assertEqual(placement.type, value.placement.type)
                test_case.assertListEqual(list(placement.ranks), list(value.placement.ranks))
                # check sbp
                test_case.assertTupleEqual(sbp, value.sbp)

    def _all_local(test_case, input):
        node_tree = IONode(value=input)
        for _, node in node_tree.named_nodes():
            if node._type == IONodeType.TENSOR:
                test_case.assertFalse(node._value.is_global)

    def test_any_input(test_case):
        tensor = flow.zeros((3, 4))
        tensor_list = [flow.tensor([1, 2, 3]), flow.randn((2, 3, 4))]
        tensor_tuple = (flow.zeros((2, 2)), flow.ones((2, 3)), flow.randn((3, 5)))
        tensor_dict = {'tensor': tensor, 'tensor_lt': tensor_list}
        random_combination = [None, 1, "test_str", tensor, tensor_list, tensor_tuple, tensor_dict]

        inputs = [None, 100, 'test_str', tensor, tensor_list, tensor_tuple, tensor_dict, random_combination]
        global_inputs = []
        for i in inputs:
            ret = flow.to_global(i, placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp)
            test_case._all_global(ret, placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp)
            global_inputs.append(ret)
        
        for i in global_inputs:
            ret = flow.to_local(i)
            test_case._all_local(ret)

    def test_tensor_to_global(test_case):
        local_tensor = flow.ones((3, 4))

        # local tensor -> global tensor
        global_tensor = flow.to_global(local_tensor, placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp)
        test_case.assertTrue(global_tensor.is_global)

        # global tensor -> global tensor
        global_tensor = flow.to_global(global_tensor, placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp)
        test_case.assertTrue(global_tensor.is_global)

        # passing no placement and sbp
        with test_case.assertRaises(ValueError):
            global_tensor = flow.to_global(local_tensor, placement=None, sbp=None)

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
        test_case._all_global(global_state_dict, placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp)

        # global state dict -> global state dict
        global_state_dict = flow.to_global(global_state_dict,
                                           placement=TestToGlobalLocal.placement,
                                           sbp=TestToGlobalLocal.sbp)
        test_case._all_global(global_state_dict, placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp)

    def _test_state_dict_to_local(test_case, global_state_dict):
        # global state dict -> local state dict
        local_state_dict = flow.to_local(global_state_dict)
        test_case._all_local(local_state_dict)
        
        # local input, display warning
        local_state_dict = flow.to_local(local_state_dict)

    def test_eagar_state_dict(test_case):
        test_case._test_state_dict_to_global(TestToGlobalLocal.model.state_dict())
        global_model = TestToGlobalLocal.model.to_global(placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp)
        test_case._test_state_dict_to_local(global_model.state_dict())

    def test_graph_state_dict(test_case):
        test_case._test_state_dict_to_global(TestToGlobalLocal.local_graph_model.state_dict())
        test_case._test_state_dict_to_local(TestToGlobalLocal.global_graph_model.state_dict())


if __name__ == "__main__":
    # test on three types of sbp
    sbp_types = [(flow.sbp.broadcast,), (flow.sbp.split(0),), (flow.sbp.partial_sum,)]
    for sbp in sbp_types:
        TestToGlobalLocal.sbp = sbp
        TestToGlobalLocal.global_graph_model = GraphModel(TestToGlobalLocal.model.to_global(placement=TestToGlobalLocal.placement, sbp=sbp))

        suite = unittest.TestSuite()
        suite.addTests([
            TestToGlobalLocal('test_any_input'),
            TestToGlobalLocal('test_tensor_to_global'),
            TestToGlobalLocal('test_tensor_to_local'),
            TestToGlobalLocal('test_eagar_state_dict'),
            TestToGlobalLocal('test_graph_state_dict')
        ])
        runner = unittest.TextTestRunner()
        runner.run(suite)
