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
import numpy as np
import sys
import oneflow.typing as oft

flow.config.gpu_device_num(4)

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.default_logical_view(flow.scope.consistent_view())

if __name__ == "__main__":

    @flow.global_function(function_config=func_config)
    def test_job(x: oft.Numpy.Placeholder((10000,), dtype=flow.float)):
        return flow.eager_nccl_all_reduce(
            x, parallel_conf="""  device_tag: "gpu", device_name: "0:0-3" """,
        )

    for _ in range(10):
        x = np.random.rand(10000).astype(np.float32)
        y = test_job(x).get()
        print(x)
        print(y)
