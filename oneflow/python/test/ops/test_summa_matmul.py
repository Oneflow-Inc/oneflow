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


def summa_broadcast_matmul(a, b, name):
    return (
        flow.user_op_builder(name)
        .Op("summa_broadcast_matmul")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("alpha", float(1))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def summa_broadcast_matmul_grad_a(a, b, name):
    return (
        flow.user_op_builder(name)
        .Op("summa_broadcast_matmul_grad_a")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("alpha", float(1))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def summa_broadcast_matmul_grad_b(a, b, name):
    return (
        flow.user_op_builder(name)
        .Op("summa_broadcast_matmul_grad_b")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("alpha", float(1))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def summa_matmul(a, b, trans_a, trans_b, name):
    return (
        flow.user_op_builder(name)
        .Op("summa_matmul")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("transpose_a", trans_a)
        .Attr("transpose_b", trans_b)
        .Attr("alpha", float(1))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


flow.config.gpu_device_num(4)
# flow.config.nccl_use_compute_stream(True)

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
b = 12
m = 64
n = 1024
k = 128
bmm_a_shape = (b, m, k)
bmm_b_shape = (k, n)

bmm_a_grad_a_shape = (b, m, n)
bmm_a_grad_b_shape = (k, n)

bmm_b_grad_a_shape = (b, m, k)
bmm_b_grad_b_shape = (b, m, n)

a_shape = (m, k)
b_shape = (k, n)
a_trans_shape = (k, m)
b_trans_shape = (n, k)

mm_a_grad_a_shape = (m, n)
mm_a_grad_b_shape = (k, n)
mm_b_grad_a_shape = (m, k)
mm_b_grad_b_shape = (m, n)
# num_nodes=1
# node_ips=["192.168.1.15", "192.168.1.16"]
# if num_nodes > 1:
#    assert num_nodes <= len(node_ips)
#    flow.env.ctrl_port(12138)
#    nodes = []
#    for ip in node_ips:
#        addr_dict = {}
#        addr_dict["addr"] = ip
#        nodes.append(addr_dict)
#    flow.env.machine(nodes)


@flow.global_function(type="predict", function_config=func_config)
def FlowJob1(
    a: oft.Numpy.Placeholder(bmm_a_shape, dtype=flow.float),
    b: oft.Numpy.Placeholder(bmm_b_shape, dtype=flow.float),
):
    with flow.scope.placement("gpu", "0:0-3", (4, 1)):
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(2)"])
        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(1)"])
        out = summa_broadcast_matmul(a, b, "summa")
    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
    return out


@flow.global_function(type="predict", function_config=func_config)
def FlowJob2(
    a: oft.Numpy.Placeholder(bmm_a_grad_a_shape, dtype=flow.float),
    b: oft.Numpy.Placeholder(bmm_a_grad_b_shape, dtype=flow.float),
):
    with flow.scope.placement("gpu", "0:0-3", (4, 1)):
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(2)"])
        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(1)"])
        out = summa_broadcast_matmul_grad_a(a, b, "summa")
    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
    return out


@flow.global_function(type="predict", function_config=func_config)
def FlowJob3(
    a: oft.Numpy.Placeholder(bmm_b_grad_a_shape, dtype=flow.float),
    b: oft.Numpy.Placeholder(bmm_b_grad_b_shape, dtype=flow.float),
):
    with flow.scope.placement("gpu", "0:0-3", (4, 1)):
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(2)"])
        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(2)"])
        out = summa_broadcast_matmul_grad_b(a, b, "summa")
    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
    return out


@flow.global_function(type="predict", function_config=func_config)
def FlowJob4(
    a: oft.Numpy.Placeholder(a_shape, dtype=flow.float),
    b: oft.Numpy.Placeholder(b_shape, dtype=flow.float),
):
    with flow.scope.placement("gpu", "0:0-3", (4, 1)):
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(1)"])
        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(1)"])
        out = summa_matmul(a, b, False, False, "summa")
    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
    return out


@flow.global_function(type="predict", function_config=func_config)
def FlowJob5(
    a: oft.Numpy.Placeholder(mm_a_grad_a_shape, dtype=flow.float),
    b: oft.Numpy.Placeholder(mm_a_grad_b_shape, dtype=flow.float),
):
    with flow.scope.placement("gpu", "0:0-3", (4, 1)):
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(1)"])
        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(1)"])
        out = summa_matmul(a, b, False, True, "summa")
    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
    return out


@flow.global_function(type="predict", function_config=func_config)
def FlowJob6(
    a: oft.Numpy.Placeholder(mm_b_grad_a_shape, dtype=flow.float),
    b: oft.Numpy.Placeholder(mm_b_grad_b_shape, dtype=flow.float),
):
    with flow.scope.placement("gpu", "0:0-3", (4, 1)):
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(1)"])
        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(1)"])
        out = summa_matmul(a, b, True, False, "summa")
    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
    return out


"""
bmm_a_shape = (b, m, k)
bmm_b_shape = (k, n)

bmm_a_grad_a_shape = (b, m, n)
bmm_a_grad_b_shape = (k, n)

bmm_b_grad_a_shape = (b, m, k)
bmm_b_grad_b_shape = (b, m, n)

a_shape = (m, k)
b_shape = (k, n)
a_trans_shape = (k, m)
b_trans_shape = (n, k)

mm_a_grad_a_shape = (m, n)
mm_a_grad_b_shape = (k, n)
mm_b_grad_a_shape = (m, k)
mm_b_grad_b_shape = (m, n)

"""

bmm_a = np.random.randn(*bmm_a_shape).astype(np.float32)
bmm_b = np.random.randn(*bmm_b_shape).astype(np.float32)

bmm_a_grad_a = np.random.randn(*bmm_a_grad_a_shape).astype(np.float32)
bmm_a_grad_b = np.random.randn(*bmm_a_grad_b_shape).astype(np.float32)

bmm_b_grad_a = np.random.randn(*bmm_b_grad_a_shape).astype(np.float32)
bmm_b_grad_b = np.random.randn(*bmm_b_grad_b_shape).astype(np.float32)


mm_a = np.random.randn(*a_shape).astype(np.float32)
mm_b = np.random.randn(*b_shape).astype(np.float32)

mm_a_grad_a = np.random.randn(*mm_a_grad_a_shape).astype(np.float32)
mm_a_grad_b = np.random.randn(*mm_a_grad_b_shape).astype(np.float32)

mm_b_grad_a = np.random.randn(*mm_b_grad_a_shape).astype(np.float32)
mm_b_grad_b = np.random.randn(*mm_b_grad_b_shape).astype(np.float32)

for i in range(1):

    c1 = FlowJob1(bmm_a, bmm_b).get().numpy()
    np_c1 = np.matmul(bmm_a, bmm_b)
    diff1 = np_c1 - c1
    print(diff1.max())
    print(diff1.min())

    c2 = FlowJob2(bmm_a_grad_a, bmm_a_grad_b).get().numpy()
    np_c2 = np.matmul(bmm_a_grad_a, bmm_a_grad_b.transpose(1, 0))
    diff2 = np_c2 - c2
    print(diff2.max())
    print(diff2.min())

    c3 = FlowJob3(bmm_b_grad_a, bmm_b_grad_b).get().numpy()
    np_c3 = np.matmul(
        bmm_b_grad_a.transpose(2, 0, 1).reshape(k, b * m),
        bmm_b_grad_b.reshape(b * m, n),
    )
    diff3 = np_c3 - c3
    print(diff3.max())
    print(diff3.min())

    c4 = FlowJob4(mm_a, mm_b).get().numpy()
    np_c4 = np.matmul(mm_a, mm_b)
    diff4 = np_c4 - c4
    print(diff4.max())
    print(diff4.min())

    c5 = FlowJob5(mm_a_grad_a, mm_a_grad_b).get().numpy()
    np_c5 = np.matmul(mm_a_grad_a, mm_a_grad_b.transpose(1, 0))
    diff5 = np_c5 - c5
    print(diff5.max())
    print(diff5.min())

    c6 = FlowJob6(mm_b_grad_a, mm_b_grad_b).get().numpy()
    np_c6 = np.matmul(mm_b_grad_a.transpose(1, 0), mm_b_grad_b)
    diff6 = np_c6 - c6
    print(diff6.max())
    print(diff6.min())
