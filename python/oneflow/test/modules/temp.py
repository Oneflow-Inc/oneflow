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
import os

import numpy as np

import oneflow as flow
from oneflow import nn
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList
from oneflow.framework.tensor import Tensor
from oneflow.nn.graph.util import ArgsTree


class GraphTestModel(nn.Graph):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def build(self, x):
        return self.model(x)


@flow.unittest.skip_unless_1n2d()
class TestToGlobalAndLocal(flow.unittest.TestCase):
    def to_global_local_test(test_case):

        placement = flow.placement("cpu", ranks=[0, 1])
        sbp = None
        model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
        local_graph_model = GraphTestModel(model)
        global_graph_model = None

        print("teststop")
        sbp_types = [
            (flow.sbp.broadcast,),
            (flow.sbp.split(0),),
            (flow.sbp.partial_sum,),
        ]
        for sbp in sbp_types:
            TestToGlobalLocal.sbp = sbp
            TestToGlobalLocal.global_graph_model = GraphTestModel(
                TestToGlobalLocal.model.to_global(
                    placement=TestToGlobalLocal.placement, sbp=sbp
                )
            )
            test_case._test_any_input()
            test_case._test_tensor_to_global()
            test_case._test_tensor_to_local()
            test_case._test_eagar_state_dict()
            test_case._test_graph_state_dict()
            test_case.assertEqual(1, 0)

    def __all_global(test_case, input, placement, sbp):
        if type(input) == Tensor:
            test_case.assertTrue(input.is_global)
            # check placement
            test_case.assertEqual(placement.type, input.placement.type)
            test_case.assertListEqual(
                list(placement.ranks), list(input.placement.ranks)
            )
            # check sbp
            test_case.assertTupleEqual(sbp, input.sbp)
        elif isinstance(input, (dict, tuple, list)):
            node_tree = ArgsTree(input)
            for node in node_tree.iter_nodes():
                if isinstance(node, Tensor):
                    test_case.assertTrue(node.is_global)
                    # check placement
                    test_case.assertEqual(placement.type, node.placement.type)
                    test_case.assertListEqual(
                        list(placement.ranks), list(node.placement.ranks)
                    )
                    # check sbp
                    test_case.assertTupleEqual(sbp, node.sbp)

    def __all_local(test_case, input):
        if type(input) == Tensor:
            test_case.assertFalse(input.is_global)
        elif isinstance(input, (dict, tuple, list)):
            node_tree = ArgsTree(input)
            for node in node_tree.iter_nodes():
                if isinstance(node, Tensor):
                    test_case.assertFalse(node.is_global)

    def _test_any_input(test_case):
        tensor = flow.zeros((3, 4))
        tensor_list = [flow.tensor([1, 2, 3]), flow.randn((2, 3, 4))]
        tensor_tuple = (flow.zeros((2, 2)), flow.ones((2, 3)), flow.randn((3, 5)))
        tensor_dict = {"tensor": tensor, "tensor_lt": tensor_list}
        random_combination = [
            None,
            1,
            "test_str",
            tensor,
            tensor_list,
            tensor_tuple,
            tensor_dict,
        ]

        inputs = [
            None,
            100,
            "test_str",
            tensor,
            tensor_list,
            tensor_tuple,
            tensor_dict,
            random_combination,
        ]
        global_inputs = []
        for i in inputs:
            ret = flow.to_global(
                i, placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp
            )
            test_case.__all_global(
                ret, placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp
            )
            global_inputs.append(ret)

        for i in global_inputs:
            ret = flow.to_local(i)
            test_case.__all_local(ret)

    def _test_tensor_to_global(test_case):
        local_tensor = flow.ones((3, 4))

        # local tensor -> global tensor
        global_tensor = flow.to_global(
            local_tensor,
            placement=TestToGlobalLocal.placement,
            sbp=TestToGlobalLocal.sbp,
        )
        test_case.assertTrue(global_tensor.is_global)

        # global tensor -> global tensor
        global_tensor = flow.to_global(
            global_tensor,
            placement=TestToGlobalLocal.placement,
            sbp=TestToGlobalLocal.sbp,
        )
        test_case.assertTrue(global_tensor.is_global)

        # passing no placement and sbp
        with test_case.assertRaises(ValueError):
            global_tensor = flow.to_global(local_tensor, placement=None, sbp=None)

        # wrong sbp type
        with test_case.assertRaises(TypeError):
            global_tensor = flow.to_global(
                local_tensor,
                placement=TestToGlobalLocal.placement,
                sbp=(TestToGlobalLocal.sbp, 0),
            )

    def _test_tensor_to_local(test_case):
        # global tensor -> local tensor
        global_tensor = flow.ones(
            (3, 4), placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp
        )
        local_tensor = flow.to_local(global_tensor)
        test_case.assertFalse(local_tensor.is_global)

    def __test_state_dict_to_global(test_case, local_state_dict):
        # local state dict -> global state dict
        global_state_dict = flow.to_global(
            local_state_dict,
            placement=TestToGlobalLocal.placement,
            sbp=TestToGlobalLocal.sbp,
        )
        test_case.__all_global(
            global_state_dict,
            placement=TestToGlobalLocal.placement,
            sbp=TestToGlobalLocal.sbp,
        )

        # global state dict -> global state dict
        global_state_dict = flow.to_global(
            global_state_dict,
            placement=TestToGlobalLocal.placement,
            sbp=TestToGlobalLocal.sbp,
        )
        test_case.__all_global(
            global_state_dict,
            placement=TestToGlobalLocal.placement,
            sbp=TestToGlobalLocal.sbp,
        )

    def __test_state_dict_to_local(test_case, global_state_dict):
        # global state dict -> local state dict
        local_state_dict = flow.to_local(global_state_dict)
        test_case.__all_local(local_state_dict)

        # local input, display warning
        local_state_dict = flow.to_local(local_state_dict)

    def _test_eagar_state_dict(test_case):
        test_case.__test_state_dict_to_global(TestToGlobalLocal.model.state_dict())
        global_model = TestToGlobalLocal.model.to_global(
            placement=TestToGlobalLocal.placement, sbp=TestToGlobalLocal.sbp
        )
        test_case.__test_state_dict_to_local(global_model.state_dict())

    def _test_graph_state_dict(test_case):
        test_case.__test_state_dict_to_global(
            TestToGlobalLocal.local_graph_model.state_dict()
        )
        test_case.__test_state_dict_to_local(
            TestToGlobalLocal.global_graph_model.state_dict()
        )


if __name__ == "__main__":
    unittest.main()
