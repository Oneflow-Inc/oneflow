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
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft
import test_global_storage
from test_util import GenArgList, type_name_to_flow_type


def embedding_prefetch(indices, embedding_size, name, embedding_name):
    num_unique_indices, unique_indices, reverse_idx = (
        flow.user_op_builder(name)
        .Op("embedding_prefetch")
        .Input("indices", [indices])
        .Output("num_unique_indices")
        .Output("unique_indices")
        .Output("reverse_idx")
        .Attr("name", embedding_name)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )
    return num_unique_indices, unique_indices, reverse_idx


def embedding_lookup(indices, embedding_size, name, embedding_name, optimizer):
    num_unique_indices, unique_indices, reverse_idx = embedding_prefetch(
        indices, embedding_size, name + "_prefetch", embedding_name
    )
    return (
        flow.user_op_builder(name)
        .Op("embedding_lookup")
        .Input("num_unique_indices", [num_unique_indices])
        .Input("unique_indices", [unique_indices])
        .Input("reverse_idx", [reverse_idx])
        .Output("embeddings")
        .Output("unique_values")
        .Attr("name", embedding_name)
        .Attr("optimizer", optimizer)
        .Attr("embedding_size", embedding_size)
        .Attr("dtype", flow.float)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


embedding_size = 128


def compare_with_tensorflow(device_type, x_shape, indices_shape, data_type):
    assert device_type in ["gpu", "cpu"]
    flow.clear_default_session()
    func_config = flow.FunctionConfig()
    func_config.default_data_type(flow.float)

    @flow.global_function(type="train", function_config=func_config)
    def EmbeddingJob(indices: oft.Numpy.Placeholder(indices_shape, dtype=flow.int64)):
        with flow.scope.placement(device_type, "0:0"):
            x = flow.get_variable(
                "x",
                shape=x_shape,
                dtype=type_name_to_flow_type[data_type],
                initializer=flow.zeros_initializer(),
                trainable=True,
            )
            x1 = embedding_lookup(
                indices,
                embedding_size=embedding_size,
                name="EmbeddingLookup1",
                embedding_name="embedding1",
                optimizer="sgd",
            )
            print("x1", x1.shape)
            loss = x + x1
            flow.optimizer.SGD(
                flow.optimizer.PiecewiseConstantScheduler([], [1]), momentum=0
            ).minimize(loss)

            return loss

    # OneFlow
    indices = np.fromfile("/data/embedding_test/bin/0.bin", dtype=np.int64)[
        0 : 16384 * 26
    ]
    indices = indices.reshape(16384, 26)
    # indices = np.random.randint(0, 51000, size=(indices_shape)).astype(np.int64)
    # np.save("indices", indices)
    of_out = EmbeddingJob(indices).get()
    print("of_out1", of_out.numpy().flatten()[0:20])
    indices = np.fromfile("/data/embedding_test/bin/0.bin", dtype=np.int64)[
        16384 * 26 : 16384 * 26 * 2
    ]
    indices = indices.reshape(16384, 26)
    of_out = EmbeddingJob(indices).get()
    print("of_out2", of_out.numpy().flatten()[0:20])
    # a, b = np.unique(indices, return_inverse=True)
    # print("a", a.size, a)
    # print("inverse_id", b.size, b)
    # np.save("indices", indices)


@flow.unittest.skip_unless_1n1d()
class TestL2Normalize(flow.unittest.TestCase):
    def test_l2_normalize(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [(16384, 26, 128)]
        arg_dict["indices_shape"] = [(16384, 26)]
        arg_dict["data_type"] = ["float32"]
        for arg in GenArgList(arg_dict):
            compare_with_tensorflow(*arg)


if __name__ == "__main__":
    unittest.main()
