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

import oneflow as flow
import oneflow.unittest
from oneflow.test_utils.test_util import (
    GenArgList,
    type_name_to_flow_type,
    type_name_to_np_type,
)


def _test_sparse_softmax_cross_entropy(
    device_type, data_type, label_type, batch_size, num_classes,
):
    data_type = type_name_to_flow_type[data_type]
    label_type = type_name_to_flow_type[label_type]
    np_labels = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
    np_logits = np.random.random((batch_size, num_classes)).astype(np.float32)

    of_logits_cpu = flow.tensor(np_logits, dtype=data_type, requires_grad=True)
    of_labels_cpu = flow.tensor(np_labels, dtype=label_type)
    of_output_cpu = flow.nn.functional.cross_entropy(
        of_logits_cpu, of_labels_cpu, reduction="none"
    )
    # of_output_cpu.sum().backward()

    of_logits_mlu = flow.tensor(
        np_logits, device=device_type, dtype=data_type, requires_grad=True
    )
    of_labels_mlu = flow.tensor(np_labels, device=device_type, dtype=label_type)
    of_output_mlu = flow.nn.functional.sparse_softmax_cross_entropy(
        labels=of_labels_mlu, logits=of_logits_mlu
    )
    # of_output.sum().backward()

    assert np.allclose(
        of_output_mlu.cpu().numpy(), of_output_cpu.numpy(), rtol=1e-03, atol=1e-04
    )
    # assert np.allclose(
    #     of_logits_mlu.grad.numpy(), of_logits_cpu.grad, rtol=1e-03, atol=1e-04
    # )


class TestSparseSoftmaxCrossEntropyWithLogits(flow.unittest.TestCase):
    @flow.unittest.skip_unless_1n1d()
    def test_sparse_softmax_cross_entropy(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["mlu"]
        arg_dict["data_type"] = ["float32"]
        arg_dict["label_type"] = ["int32"]
        arg_dict["batch_size"] = [64, 16]
        arg_dict["num_classes"] = [100, 1000]
        for arg in GenArgList(arg_dict):
            _test_sparse_softmax_cross_entropy(*arg)


if __name__ == "__main__":
    unittest.main()
