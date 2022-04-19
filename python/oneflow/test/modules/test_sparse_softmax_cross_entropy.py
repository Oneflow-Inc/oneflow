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
from oneflow.test_utils.test_util import (
    GenArgList,
    type_name_to_flow_type,
    type_name_to_np_type,
)


def compare_with_torch(
    device_type, data_type, label_type, batch_size, num_classes,
):
    data_type = type_name_to_flow_type[data_type]
    label_type = type_name_to_flow_type[label_type]
    np_labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
    np_logits = np.random.random((batch_size, num_classes)).astype(np.float32)

    torch_logits = torch.tensor(np_logits, dtype=torch.float32, requires_grad=True)
    torch_labels = torch.tensor(np_labels, dtype=torch.int64)
    torch_output = torch.nn.functional.cross_entropy(
        torch_logits, torch_labels, reduction="none"
    )
    torch_output.sum().backward()

    of_logits = flow.tensor(
        np_logits, device=device_type, dtype=data_type, requires_grad=True
    )
    of_labels = flow.tensor(np_labels, device=device_type, dtype=label_type)
    of_output = flow.nn.functional.sparse_softmax_cross_entropy(
        labels=of_labels, logits=of_logits
    ).to(device_type)
    of_output.sum().backward()

    assert np.allclose(
        of_output.numpy(), torch_output.detach().numpy(), rtol=1e-03, atol=1e-04
    )
    assert np.allclose(
        of_logits.grad.numpy(), torch_logits.grad, rtol=1e-03, atol=1e-04
    )


def compare_eager_global_with_torch(
    device_type, data_type, label_type, batch_size, num_classes,
):
    data_type = type_name_to_flow_type[data_type]
    label_type = type_name_to_flow_type[label_type]
    np_labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
    np_logits = np.random.random((batch_size, num_classes)).astype(np.float32)
    placement = flow.placement(device_type, range(4))
    rank = flow.env.get_rank()
    if rank == 0:
        torch_logits = torch.tensor(np_logits, dtype=torch.float32, requires_grad=True)
        torch_labels = torch.tensor(np_labels, dtype=torch.int64)
        torch_output = torch.nn.functional.cross_entropy(
            torch_logits, torch_labels, reduction="none"
        )
        torch_output.sum().backward()

    # 1D sbp
    of_logits = flow.tensor(
        np_logits, device=device_type, dtype=data_type, requires_grad=True
    )
    flow.comm.broadcast(of_logits, 0)
    of_logits = of_logits.to_global(placement=placement, sbp=[flow.sbp.broadcast])
    of_logits.retain_grad()
    global_of_logits = of_logits.to_global(placement=placement, sbp=[flow.sbp.split(1)])
    of_labels = flow.tensor(np_labels, device=device_type, dtype=label_type)
    flow.comm.broadcast(of_labels, 0)
    of_labels = of_labels.to_global(placement=placement, sbp=[flow.sbp.broadcast])

    of_output = flow.nn.functional.sparse_softmax_cross_entropy(
        labels=of_labels, logits=global_of_logits
    ).to(device_type)
    of_output.sum().backward()
    of_logits_grad = of_logits.grad.to_global(
        placement=placement, sbp=[flow.sbp.broadcast]
    )
    of_logits_grad = of_logits_grad.to_local()
    of_output = of_output.to_global(placement=placement, sbp=[flow.sbp.broadcast])
    of_output = of_output.to_local()

    if rank == 0:
        assert np.allclose(
            of_output.numpy(), torch_output.detach().numpy(), rtol=1e-03, atol=1e-04
        )
        assert np.allclose(
            of_logits_grad.numpy(), torch_logits.grad, rtol=1e-03, atol=1e-04
        )


def compare_eager_2d_global_with_torch(
    device_type, data_type, label_type, batch_size, num_classes,
):
    data_type = type_name_to_flow_type[data_type]
    label_type = type_name_to_flow_type[label_type]
    np_labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
    np_logits = np.random.random((batch_size, num_classes)).astype(np.float32)

    rank = flow.env.get_rank()
    if rank == 0:
        torch_logits = torch.tensor(np_logits, dtype=torch.float32, requires_grad=True)
        torch_labels = torch.tensor(np_labels, dtype=torch.int64)
        torch_output = torch.nn.functional.cross_entropy(
            torch_logits, torch_labels, reduction="none"
        )
        torch_output.sum().backward()

    # 2D sbp
    placement = flow.placement("cuda", ranks=[[0, 1], [2, 3]])
    of_logits = flow.tensor(
        np_logits, device=device_type, dtype=data_type, requires_grad=True
    )
    flow.comm.broadcast(of_logits, 0)
    of_logits = of_logits.to_global(
        placement=placement, sbp=[flow.sbp.broadcast, flow.sbp.broadcast]
    )
    of_logits.retain_grad()
    global_of_logits = of_logits.to_global(
        placement=placement, sbp=[flow.sbp.split(0), flow.sbp.split(1)]
    )
    of_labels = flow.tensor(np_labels, device=device_type, dtype=label_type)
    flow.comm.broadcast(of_labels, 0)
    of_labels = of_labels.to_global(
        placement=placement, sbp=[flow.sbp.broadcast, flow.sbp.broadcast]
    )
    of_labels = of_labels.to_global(
        placement=placement, sbp=[flow.sbp.split(0), flow.sbp.broadcast]
    )

    of_output = flow.nn.functional.sparse_softmax_cross_entropy(
        labels=of_labels, logits=global_of_logits
    ).to(device_type)
    of_output.sum().backward()
    of_logits_grad = of_logits.grad.to_global(
        placement=placement, sbp=[flow.sbp.broadcast, flow.sbp.broadcast]
    )
    of_logits_grad = of_logits_grad.to_local()
    of_output = of_output.to_global(
        placement=placement, sbp=[flow.sbp.broadcast, flow.sbp.broadcast]
    )
    of_output = of_output.to_local()

    if rank == 0:
        assert np.allclose(
            of_output.numpy(), torch_output.detach().numpy(), rtol=1e-03, atol=1e-04
        )
        assert np.allclose(
            of_logits_grad.numpy(),
            torch_logits.grad.detach().numpy(),
            rtol=1e-03,
            atol=1e-04,
        )


def compare_lazy_global_with_torch(
    device_type, data_type, label_type, batch_size, num_classes,
):
    data_type = type_name_to_flow_type[data_type]
    label_type = type_name_to_flow_type[label_type]
    np_labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
    np_logits = np.random.random((batch_size, num_classes)).astype(np.float32)
    placement = flow.placement(device_type, range(4))
    rank = flow.env.get_rank()

    if rank == 0:
        torch_logits = torch.tensor(np_logits, dtype=torch.float32, requires_grad=True)
        torch_labels = torch.tensor(np_labels, dtype=torch.int64)
        torch_output = torch.nn.functional.cross_entropy(
            torch_logits, torch_labels, reduction="none"
        )
        torch_output.sum().backward()

    class MyModule(flow.nn.Graph):
        def __init__(self):
            super(MyModule, self).__init__()

        def build(self, logits, labels):
            output = flow.nn.functional.sparse_softmax_cross_entropy(
                labels=labels, logits=logits
            )
            # nn.graph no support get input.grad
            # output.sum().backward()
            return output

    of_logits = flow.tensor(
        np_logits, device=device_type, dtype=data_type, requires_grad=True
    )
    flow.comm.broadcast(of_logits, 0)
    of_logits = of_logits.to_global(placement=placement, sbp=[flow.sbp.broadcast])
    of_logits.retain_grad()
    global_of_logits = of_logits.to_global(placement=placement, sbp=[flow.sbp.split(1)])
    of_labels = flow.tensor(np_labels, device=device_type, dtype=label_type)
    flow.comm.broadcast(of_labels, 0)
    of_labels = of_labels.to_global(placement=placement, sbp=[flow.sbp.broadcast])
    graph = MyModule()
    of_output = graph(global_of_logits, of_labels)
    of_output = of_output.to_global(placement=placement, sbp=[flow.sbp.broadcast])
    of_output = of_output.to_local()

    flow._oneflow_internal.eager.Sync()

    if rank == 0:
        assert np.allclose(
            of_output.numpy(), torch_output.detach().numpy(), rtol=1e-03, atol=1e-04
        )


class TestSparseSoftmaxCrossEntropyWithLogits(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_sparse_softmax_cross_entropy(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cuda", "cpu"]
        arg_dict["data_type"] = ["float32", "double"]
        arg_dict["label_type"] = ["int32", "int64"]
        arg_dict["batch_size"] = [64, 16]
        arg_dict["num_classes"] = [100, 1000]
        for arg in GenArgList(arg_dict):
            compare_with_torch(*arg)


class TestSparseSoftmaxCrossEntropyMsWithLogits(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    @flow.unittest.skip_unless_1n4d()
    def test_distributed_sparse_softmax_cross_entropy(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["cuda"]
        arg_dict["data_type"] = ["float32", "double"]
        arg_dict["label_type"] = ["int32", "int64"]
        arg_dict["batch_size"] = [64]
        arg_dict["num_classes"] = [1000]
        for arg in GenArgList(arg_dict):
            # compare_eager_global_with_torch(*arg)
            compare_eager_2d_global_with_torch(*arg)
            compare_lazy_global_with_torch(*arg)


if __name__ == "__main__":
    unittest.main()
