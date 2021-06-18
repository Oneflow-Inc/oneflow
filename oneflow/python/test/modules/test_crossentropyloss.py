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

import numpy as np

import oneflow.experimental as flow
from test_util import GenArgList

g_test_samples = [
    {
        "input": np.array(
            [
                [-0.6980871, 0.4765042, -1.969919, 0.28965086, -0.53548324],
                [-0.26332688, 0.27541, 0.30080616, 0.09914763, 0.53522176],
                [0.7332028, 0.38375184, -0.2831992, -0.9833142, 0.387824],
            ]
        ),
        "target": np.array([3, 3, 4], dtype=np.int32),
        "ignore_index": 4,
        "out": np.array([1.1380, 1.7332, 0.0], dtype=np.float32),
        "out_sum": np.array([2.8711782], dtype=np.float32),
        "out_mean": np.array([1.4355891], dtype=np.float32),
    },
    {
        "input": np.array(
            [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
        ),
        "target": np.array([[[1, 0], [0, 1]]], dtype=np.int32),
        "ignore_index": 1,
        "out": np.array([[[0.0, 0.6832], [0.8544, 0.0]]], dtype=np.float32),
        "out_sum": np.array([1.5375525], dtype=np.float32),
        "out_mean": np.array([0.76877624], dtype=np.float32),
    },
    {
        "input": np.array(
            [
                [-0.6980871, 0.4765042, -1.969919, 0.28965086, -0.53548324],
                [-0.26332688, 0.27541, 0.30080616, 0.09914763, 0.53522176],
                [0.7332028, 0.38375184, -0.2831992, -0.9833142, 0.387824],
            ]
        ),
        "target": np.array([3, 3, 4], dtype=np.int32),
        "out": np.array([1.1380, 1.7332, 1.4287], dtype=np.float32),
        "out_sum": np.array([4.2999], dtype=np.float32),
        "out_mean": np.array([1.4333], dtype=np.float32),
    },
    {
        "input": np.array(
            [[[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]]]
        ),
        "target": np.array([[[1, 0], [0, 1]]], dtype=np.int32),
        "out": np.array([[[0.6882, 0.6832], [0.8544, 1.8006]]], dtype=np.float32),
        "out_sum": np.array([4.0263], dtype=np.float32),
        "out_mean": np.array([1.0066], dtype=np.float32),
    },
    {
        "input": np.array(
            [
                [[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]],
                [[[0.12, 0.36], [0.22, 0.66]], [[0.13, 0.34], [0.52, -0.96]]],
            ]
        ),
        "target": np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype=np.int32),
        "out": np.array(
            [
                [[0.6882, 0.6832], [0.8544, 1.8006]],
                [[0.6882, 0.6832], [0.8544, 1.8006]],
            ],
            dtype=np.float32,
        ),
        "out_sum": np.array([8.0526], dtype=np.float32),
        "out_mean": np.array([1.0066], dtype=np.float32),
    },
    {
        "input": np.array([[[0.12, 0.36, 0.22, 0.66], [0.13, 0.34, 0.52, -0.96]]]),
        "target": np.array([[1, 0, 0, 1]], dtype=np.int32),
        "out": np.array([[0.6882, 0.6832, 0.8544, 1.8006]], dtype=np.float32,),
        "out_sum": np.array([4.0263], dtype=np.float32),
        "out_mean": np.array([1.0066], dtype=np.float32),
    },
]


@unittest.skipIf(
    not flow.unittest.env.eager_execution_enabled(),
    ".numpy() doesn't work in lazy mode",
)
class TestCrossEntropyLossModule(flow.unittest.TestCase):
    def test_CrossEntropyLoss(test_case):
        global g_test_samples
        for sample in g_test_samples:
            ignore_index = sample.get("ignore_index", None)
            input = flow.Tensor(sample["input"], dtype=flow.float32)
            target = flow.Tensor(sample["target"], dtype=flow.int32)

            loss = flow.nn.CrossEntropyLoss(reduction=None, ignore_index=ignore_index)
            of_out = loss(input, target)
            assert np.allclose(of_out.numpy(), sample["out"], 1e-4, 1e-4)

            loss_sum = flow.nn.CrossEntropyLoss(
                reduction="sum", ignore_index=ignore_index
            )
            of_out_sum = loss_sum(input, target)
            assert np.allclose(of_out_sum.numpy(), sample["out_sum"], 1e-4, 1e-4)

            loss_mean = flow.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )
            of_out_mean = loss_mean(input, target)
            assert np.allclose(of_out_mean.numpy(), sample["out_mean"], 1e-4, 1e-4)


if __name__ == "__main__":
    unittest.main()
