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
import oneflow as flow
import numpy as np
import oneflow.typing as oft


def summa_matmul_ab(a, b, name):
    return (
        flow.user_op_builder(name)
        .Op("matmul_ab")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("alpha", float(1))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


flow.config.gpu_device_num(4)
# flow.config.nccl_use_compute_stream(True)

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
m = 1024
n = 1024
k = 512
a_shape = (m, k)
a_trans_shape = (k, m)
b_shape = (k, n)
b_trans_shape = (n, k)


@flow.global_function(type="predict", function_config=func_config)
def FlowJob1(
    a: oft.Numpy.Placeholder(a_shape, dtype=flow.float),
    b: oft.Numpy.Placeholder(b_shape, dtype=flow.float),
):
    with flow.scope.placement("gpu", "0:0-3", (2, 2)):
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(1)"])
        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(1)"])
        out = summa_matmul_ab(a, b, "summa")
    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
    return out


for i in range(10):
    # test ab
    a = np.random.randn(*a_shape).astype(np.float32)
    b = np.random.randn(*b_shape).astype(np.float32)
    c1 = FlowJob1(a, b).get()
    print(c1.flatten()[0:10])
    # c2 = FlowJob2(a, b).get()
    # diff = c2 - c1
    # print(diff.max(), diff.min())
