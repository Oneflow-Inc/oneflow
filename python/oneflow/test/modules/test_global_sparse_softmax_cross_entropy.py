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
import os
from collections import OrderedDict

import numpy as np
import torch

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import GenArgList, type_name_to_flow_type

from oneflow.test_utils.automated_test_util.generators import *
from oneflow.test_utils.automated_test_util.torch_flow_dual_object import globaltest


def _compare_eager_global_with_torch(
    placement, logits_sbp, labels_sbp, data_type, label_type, batch_size, num_classes,
):
    data_type = type_name_to_flow_type[data_type]
    label_type = type_name_to_flow_type[label_type]
    np_labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
    np_logits = np.random.random((batch_size, num_classes)).astype(np.float32)
    if flow.env.get_rank() == 0:
        torch_logits = torch.tensor(np_logits, dtype=torch.float32, requires_grad=True)
        torch_labels = torch.tensor(np_labels, dtype=torch.int64)
        torch_output = torch.nn.functional.cross_entropy(
            torch_logits, torch_labels, reduction="none"
        )
        torch_output.sum().backward()

    of_logits = flow.tensor(np_logits, dtype=data_type, requires_grad=True).to_global(
        flow.placement.all("cpu"), flow.sbp.broadcast
    )
    of_logits = of_logits.to_global(placement, logits_sbp)

    of_logits.retain_grad()

    of_labels = flow.tensor(np_labels, dtype=label_type).to_global(
        flow.placement.all("cpu"), flow.sbp.broadcast
    )
    of_labels = of_labels.to_global(placement, labels_sbp)

    of_output = flow.nn.functional.sparse_softmax_cross_entropy(
        labels=of_labels, logits=of_logits
    )
    of_output.sum().backward()
    of_logits_grad = of_logits.grad.to_global(
        flow.placement.all("cpu"), flow.sbp.broadcast
    )
    of_logits_grad = of_logits_grad.to_local()
    of_output = of_output.to_global(flow.placement.all("cpu"), flow.sbp.broadcast)
    of_output = of_output.to_local()

    if flow.env.get_rank() == 0:
        assert np.allclose(
            of_output.numpy(), torch_output.detach().numpy(), rtol=1e-03, atol=1e-04
        )
        assert np.allclose(
            of_logits_grad.numpy(), torch_logits.grad, rtol=1e-03, atol=1e-04
        )


def _compare_lazy_global_with_torch(
    placement, logits_sbp, labels_sbp, data_type, label_type, batch_size, num_classes,
):
    data_type = type_name_to_flow_type[data_type]
    label_type = type_name_to_flow_type[label_type]
    np_labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
    np_logits = np.random.random((batch_size, num_classes)).astype(np.float32)
    if flow.env.get_rank() == 0:
        torch_logits = torch.tensor(np_logits, dtype=torch.float32, requires_grad=True)
        torch_labels = torch.tensor(np_labels, dtype=torch.int64)
        torch_output = torch.nn.functional.cross_entropy(
            torch_logits, torch_labels, reduction="none"
        )

    class MyModule(flow.nn.Graph):
        def __init__(self):
            super(MyModule, self).__init__()

        # nn.graph no support get input.grad
        def build(self, logits, labels):
            output = flow.nn.functional.sparse_softmax_cross_entropy(
                labels=labels, logits=logits
            )
            return output

    of_logits = flow.tensor(np_logits, dtype=data_type, requires_grad=True).to_global(
        flow.placement.all("cpu"), flow.sbp.broadcast
    )
    of_logits = of_logits.to_global(placement, logits_sbp)

    of_labels = flow.tensor(np_labels, dtype=label_type).to_global(
        flow.placement.all("cpu"), flow.sbp.broadcast
    )
    of_labels = of_labels.to_global(placement, labels_sbp)
    graph = MyModule()
    of_output = graph(of_logits, of_labels)
    of_output = of_output.to_global(
        placement=flow.placement.all("cpu"), sbp=[flow.sbp.broadcast]
    )
    of_output = of_output.to_local()

    flow._oneflow_internal.eager.multi_client.Sync()

    if flow.env.get_rank() == 0:
        assert np.allclose(
            of_output.numpy(), torch_output.detach().numpy(), rtol=1e-03, atol=1e-04
        )


class TestGlobalSparseSoftmaxCrossEntropyWithLogits(flow.unittest.TestCase):
    @globaltest
    def test_eager_global_sparse_softmax_cross_entropy(test_case):
        arg_dict = OrderedDict()
        arg_dict["data_type"] = ["float32", "double"]
        arg_dict["label_type"] = ["int32", "int64"]
        arg_dict["batch_size"] = [64]
        arg_dict["num_classes"] = [1024]
        for arg in GenArgList(arg_dict):
            for placement in all_placement():
                for logits_sbp in all_sbp(placement, max_dim=2):
                    for labels_sbp in all_sbp(placement, max_dim=1):
                        _compare_eager_global_with_torch(
                            placement, logits_sbp, labels_sbp, *arg
                        )

    # TODO: Too many streams will cause bugs, open the graph mode after solving
    # @globaltest
    # def test_lazy_global_sparse_softmax_cross_entropy(test_case):
    #     arg_dict = OrderedDict()
    #     arg_dict["data_type"] = ["float32", "double"]
    #     arg_dict["label_type"] = ["int32", "int64"]
    #     arg_dict["batch_size"] = [64]
    #     arg_dict["num_classes"] = [1024]
    #     for arg in GenArgList(arg_dict):
    #         for placement in all_placement():
    #             for logits_sbp in all_sbp(placement, max_dim=2):
    #                 for labels_sbp in all_sbp(placement, max_dim=1):
    #                     _compare_lazy_global_with_torch(
    #                         placement, logits_sbp, labels_sbp, *arg
    #                     )


if __name__ == "__main__":
    unittest.main()
