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
        .Op("summa_matmul_placeholder")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("alpha", float(1))
        .Attr("transpose_a", False)
        .Attr("transpose_b", False)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def summa_matmul_abt(a, b, name):
    return (
        flow.user_op_builder(name)
        .Op("summa_matmul_placeholder")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("alpha", float(1))
        .Attr("transpose_a", False)
        .Attr("transpose_b", True)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def summa_matmul_atb(a, b, name):
    return (
        flow.user_op_builder(name)
        .Op("summa_matmul_placeholder")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("alpha", float(1))
        .Attr("transpose_a", True)
        .Attr("transpose_b", False)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def summa_matmul_atbt(a, b, name):
    return (
        flow.user_op_builder(name)
        .Op("summa_matmul_placeholder")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("alpha", float(1))
        .Attr("transpose_a", True)
        .Attr("transpose_b", True)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


flow.config.gpu_device_num(4)
flow.config.enable_legacy_model_io()
flow.config.enable_model_io_v2(True)
flow.config.disable_group_boxing_by_dst_parallel(True)
flow.config.nccl_use_compute_stream(True)
func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
func_config.prune_parallel_cast_ops(False)
m = 1024
n = 1024
k = 512
a_shape = (m, k)
a_trans_shape = (k, m)
b_shape = (k, n)
b_trans_shape = (n, k)

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


# @flow.global_function(type="predict", function_config=func_config)
# def FlowJob1(
#    a: oft.Numpy.Placeholder(a_shape, dtype=flow.float),
#    b: oft.Numpy.Placeholder(b_shape, dtype=flow.float),
# ):
#    with flow.scope.placement("gpu", "0:0-3", (2, 2)):
#        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(0)"])
#        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(1)"])
#        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(0)"])
#        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(1)"])
#        out = summa_matmul_ab(a, b, "summa")
#    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
#    return out
#
#
# @flow.global_function(type="predict", function_config=func_config)
# def FlowJob2(
#    a: oft.Numpy.Placeholder(a_shape, dtype=flow.float),
#    b: oft.Numpy.Placeholder(b_shape, dtype=flow.float),
# ):
#    with flow.scope.placement("gpu", "0:0-3"):
#        out = flow.matmul(a, b)
#    return out


# @flow.global_function(type="predict", function_config=func_config)
# def FlowJob3(
#   a: oft.Numpy.Placeholder(a_shape, dtype=flow.float),
#   b: oft.Numpy.Placeholder(b_trans_shape, dtype=flow.float),
# ):
#   with flow.scope.placement("gpu", "0:0-3", (2, 2)):
#       a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(0)"])
#       a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(1)"])
#       b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(0)"])
#       b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(1)"])
#       out = summa_matmul_abt(a, b, "summa_trans")
#   out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
#   return out
#
#
# @flow.global_function(type="predict", function_config=func_config)
# def FlowJob4(
#   a: oft.Numpy.Placeholder(a_shape, dtype=flow.float),
#   b: oft.Numpy.Placeholder(b_trans_shape, dtype=flow.float),
# ):
#   with flow.scope.placement("gpu", "0:0-3"):
#       out = flow.matmul(a, b, transpose_b=True)
#   return out


# @flow.global_function(type="predict", function_config=func_config)
# def FlowJob5(
#    a: oft.Numpy.Placeholder(a_trans_shape, dtype=flow.float),
#    b: oft.Numpy.Placeholder(b_shape, dtype=flow.float),
# ):
#    with flow.scope.placement("gpu", "0:0-3", (2, 2)):
#        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(0)"])
#        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(1)"])
#        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(0)"])
#        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(1)"])
#        out = summa_matmul_atb(a, b, "summa_trans")
#    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
#    return out
#
#
# @flow.global_function(type="predict", function_config=func_config)
# def FlowJob6(
#    a: oft.Numpy.Placeholder(a_trans_shape, dtype=flow.float),
#    b: oft.Numpy.Placeholder(b_shape, dtype=flow.float),
# ):
#    with flow.scope.placement("gpu", "0:0-3"):
#        out = flow.matmul(a, b, transpose_a=True)
#    return out


# @flow.global_function(type="predict", function_config=func_config)
# def FlowJob7(
#    a: oft.Numpy.Placeholder(a_trans_shape, dtype=flow.float),
#    b: oft.Numpy.Placeholder(b_trans_shape, dtype=flow.float),
# ):
#    with flow.scope.placement("gpu", "0:0-3", (2, 2)):
#        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(0)"])
#        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(1)"])
#        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(0)"])
#        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(1)"])
#        out = summa_matmul_atbt(a, b, "summa_trans")
#    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
#    return out
#
#
# @flow.global_function(type="predict", function_config=func_config)
# def FlowJob8(
#    a: oft.Numpy.Placeholder(a_trans_shape, dtype=flow.float),
#    b: oft.Numpy.Placeholder(b_trans_shape, dtype=flow.float),
# ):
#    with flow.scope.placement("gpu", "0:0-3"):
#        out = flow.matmul(a, b, transpose_a=True, transpose_b=True)
#    return out


@flow.global_function(type="train", function_config=func_config)
def FlowJobTrain(
    a: oft.Numpy.Placeholder(a_trans_shape, dtype=flow.float),
    b: oft.Numpy.Placeholder(b_shape, dtype=flow.float),
):
    with flow.scope.placement("gpu", "0:0-3", (2, 2)):
        v1 = flow.get_variable(
            "a",
            shape=a_trans_shape,
            dtype=flow.float,
            initializer=flow.zeros_initializer(),  # flow.random_uniform_initializer(minval=0, maxval=1),
            trainable=True,
            parallel_distribution=["S(0)", "S(1)"],
        )
        v2 = flow.get_variable(
            "b",
            shape=b_shape,
            dtype=flow.float,
            initializer=flow.zeros_initializer(),  # flow.random_uniform_initializer(minval=0, maxval=1),
            trainable=True,
            parallel_distribution=["S(0)", "S(1)"],
        )
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(0)"])
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(1)"])
        a = a + v1
        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(0)"])
        b = flow.hierarchical_parallel_cast(b, parallel_distribution=["S(0)", "S(1)"])
        b = b + v2
        out = summa_matmul_atb(a, b, "summa_trans")
        out = flow.hierarchical_parallel_cast(
            out, parallel_distribution=["S(0)", "S(0)"]
        )
    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
    flow.optimizer.SGD(
        flow.optimizer.PiecewiseConstantScheduler([], [1]), momentum=0
    ).minimize(out)
    return out


check_point = flow.train.CheckPoint()
check_point.init()

for i in range(10):
    # test ab
    # a = np.random.randn(*a_shape).astype(np.float32)
    # b = np.random.randn(*b_shape).astype(np.float32)
    # c1 = FlowJob1(a, b).get()
    # c2 = FlowJob2(a, b).get()
    # diff = c2 - c1
    # print(diff.max(), diff.min())

    # test abt
    # a = np.random.randn(*a_shape).astype(np.float32)
    # b = np.random.randn(*b_trans_shape).astype(np.float32)
    # c3 = FlowJob3(a, b).get()
    # c4 = FlowJob4(a, b).get()
    # diff = c4 - c3
    # print(diff.max(), diff.min())

    # test atb
    # a = np.random.randn(*a_trans_shape).astype(np.float32)
    # b = np.random.randn(*b_shape).astype(np.float32)
    # c5 = FlowJob5(a, b).get()
    # c6 = FlowJob6(a, b).get()
    # diff = c6 - c5
    # print(diff.max(), diff.min())

    # test atb
    # a = np.random.randn(*a_trans_shape).astype(np.float32)
    # b = np.random.randn(*b_trans_shape).astype(np.float32)
    # c7 = FlowJob7(a, b).get()
    # c8 = FlowJob8(a, b).get()
    # print(c7.numpy())
    # print(c8.numpy())
    # diff = c8 - c7
    # print(diff.max(), diff.min())
    a = np.load("a_trans.npy")
    b = np.load("b.npy")
    # a = np.random.randn(*a_trans_shape).astype(np.float32)
    # b = np.random.randn(*b_shape).astype(np.float32)
    c = FlowJobTrain(a, b).get()
    print(c.numpy().flatten()[0:10])
