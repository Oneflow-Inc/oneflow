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
        .Op("summa_matmul_ab")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("alpha", float(1))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def summa_matmul_ab_no_pipeline(a, b, name):
    return (
        flow.user_op_builder(name)
        .Op("summa_matmul_ab_no_pipeline")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("alpha", float(1))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def summa_matmul_abt(a, b, name):
    return (
        flow.user_op_builder(name)
        .Op("summa_matmul_abt")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("alpha", float(1))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


def summa_matmul_atb(a, b, name):
    return (
        flow.user_op_builder(name)
        .Op("summa_matmul_atb")
        .Input("a", [a])
        .Input("b", [b])
        .Output("out")
        .Attr("alpha", float(1))
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


flow.config.gpu_device_num(4)

flow.config.enable_legacy_model_io()
flow.config.enable_model_io_v2(True)
# flow.config.nccl_use_compute_stream(True)

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)
m = 1024
n = 1024
k = 1024
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


@flow.global_function(type="predict", function_config=func_config)
def FlowJobABNopipeline(a: oft.Numpy.Placeholder(a_shape, dtype=flow.float),):
    with flow.scope.placement("gpu", "0:0-3", (2, 2)):
        b = flow.get_variable(
            name="b",
            shape=b_shape,
            parallel_distribution=["S(0)", "S(1)"],
            initializer=flow.ones_initializer(),
        )
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(1)"])
        out = summa_matmul_ab_no_pipeline(a, b, "summa0")
        out = summa_matmul_ab_no_pipeline(out, flow.identity(b), "summa1")
        # out = summa_matmul_ab_no_pipeline(out, b, "summa2")
        # out = summa_matmul_ab_no_pipeline(out, b, "summa3")
        # out = summa_matmul_ab_no_pipeline(out, b, "summa4")
    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
    return out


@flow.global_function(type="predict", function_config=func_config)
def FlowJobAB(a: oft.Numpy.Placeholder(a_shape, dtype=flow.float),):
    with flow.scope.placement("gpu", "0:0-3", (2, 2)):
        b = flow.get_variable(
            name="b0",
            shape=b_shape,
            parallel_distribution=["S(0)", "S(1)"],
            initializer=flow.ones_initializer(),
        )
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "S(1)"])
        out = summa_matmul_ab(a, b, "summa")
    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
    return out


@flow.global_function(type="predict", function_config=func_config)
def FlowJobAB1D(a: oft.Numpy.Placeholder(a_shape, dtype=flow.float),):
    with flow.scope.placement("gpu", "0:0-3"):
        b = flow.get_variable(
            name="b1", shape=b_shape, initializer=flow.ones_initializer(),
        )
        out = flow.matmul(a, b)
    return out


@flow.global_function(type="predict", function_config=func_config)
def FlowJobAB2D(a: oft.Numpy.Placeholder(a_shape, dtype=flow.float),):
    with flow.scope.placement("gpu", "0:0-3", (2, 2)):
        b = flow.get_variable(
            name="b2",
            shape=b_shape,
            parallel_distribution=["B", "S(1)"],
            initializer=flow.ones_initializer(),
        )
        a = flow.hierarchical_parallel_cast(a, parallel_distribution=["S(0)", "B"])
        out = flow.matmul(a, b)
    out = flow.hierarchical_parallel_cast(out, parallel_distribution=["S(0)"])
    return out


check_point = flow.train.CheckPoint()
check_point.init()

for i in range(10):

    # test ab pipeline
    a = np.random.randn(*a_shape).astype(np.float32)
    out = FlowJobABNopipeline(a).get()
    print(out.numpy().flatten()[0:10])

    # test ab
    # a = np.random.randn(*a_shape).astype(np.float32)
    # out = FlowJobAB(a).get()
    # print(out.numpy().flatten()[0:10])

    # test ab 2D
    # a = np.random.randn(*a_shape).astype(np.float32)
    # out = FlowJobAB2D(a).get()
    # print(out.numpy().flatten()[0:10])

    # test ab 1D
    # a = np.random.randn(*a_shape).astype(np.float32)
    # out = FlowJobAB1D(a).get()
    # print(out.numpy().flatten()[0:10])
