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
import numpy as np
import oneflow as flow
import oneflow.typing as oft
import unittest
import os
from collections import OrderedDict
from test_util import GenArgList, type_name_to_flow_type, type_name_to_np_type


func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)


def partition(input, in_num_unique, parallel_num, num_classes, name):
    return (
        flow.user_op_builder(name)
        .Op("partition")
        .Input("in", [input])
        .Input("in_size", [in_num_unique])
        .Output("out", parallel_num)
        .Output("out_size", parallel_num)
        .Attr("parallel_num", parallel_num)
        .Attr("num_classes", num_classes)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()
    )


def _run_test(test_case, device, x_shape, num_classes, parallel_num, dtype):
    num_classes_per_rank = num_classes / parallel_num
    flow.clear_default_session()

    @flow.global_function(function_config=func_config)
    def PartitionJob(
        x: oft.Numpy.Placeholder(x_shape, dtype=type_name_to_flow_type[dtype])
    ):
        with flow.scope.placement(device, "0:0"):
            y, idx, count, num_unique = flow.experimental.unique_with_counts(x)
            out_list = partition(y, num_unique, parallel_num, num_classes, "partition")
            partition_out_list = out_list[0:parallel_num]
            out_size_list = out_list[parallel_num:]
            result_list = []
            for i in range(parallel_num):
                result = flow.sync_dynamic_resize(
                    partition_out_list[i], out_size_list[i]
                )
                result_list.append(result)
            return result_list

    x = np.random.randint(0, num_classes, size=(x_shape)).astype(
        type_name_to_np_type[dtype]
    )
    result_list = PartitionJob(x).get()
    unique_x = np.unique(x)
    for i in range(parallel_num):
        lower = i * num_classes_per_rank
        upper = (i + 1) * num_classes_per_rank
        condition = (unique_x >= lower) & (unique_x < upper)
        y = unique_x[condition] - lower
        assert np.array_equal(y, result_list[i].numpy_list()[0])


@flow.unittest.skip_unless_1n1d()
class TestPartition(flow.unittest.TestCase):
    @unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
    def test_partition_gpu(test_case):
        arg_dict = OrderedDict()
        arg_dict["device_type"] = ["gpu"]
        arg_dict["x_shape"] = [
            (60,),
        ]
        arg_dict["num_classes"] = [3200]
        arg_dict["parallel_num"] = [20]
        arg_dict["dtype"] = ["int64"]
        for arg in GenArgList(arg_dict):
            _run_test(test_case, *arg)


if __name__ == "__main__":
    unittest.main()
