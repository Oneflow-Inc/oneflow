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
from __future__ import absolute_import

import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.distribute as distribute_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
from oneflow.python.oneflow_export import oneflow_export
import oneflow
import oneflow_api
from typing import Union, Optional, Sequence


@oneflow_export("repeat")
def api_repeat(
    input: oneflow_api.BlobDesc, repeat_num: int, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    func = enable_if.unique([repeat])
    return func(input, repeat_num, name=name)


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def repeat(input, repeat_num, name=None):
    assert not oneflow.eager_execution_enabled()
    return (
        oneflow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Repeat_")
        )
        .Op("repeat")
        .Input("in", [input])
        .Output("out")
        .Attr("repeat_num", repeat_num)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("acc")
def api_acc(
    one: oneflow_api.BlobDesc, max_acc_num: int, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    func = enable_if.unique([acc])
    return func(one, max_acc_num, name=name)


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def acc(one, max_acc_num, name=None):
    assert not oneflow.eager_execution_enabled()
    return (
        oneflow.user_op_builder(name if name is not None else id_util.UniqueStr("Acc_"))
        .Op("acc")
        .Input("in", [one])
        .Output("out")
        .Attr("max_acc_num", max_acc_num)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("unpack")
def api_unpack(
    input: oneflow_api.BlobDesc, unpack_num: int, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    func = enable_if.unique([unpack])
    return func(input, unpack_num, name=name)


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def unpack(input, unpack_num, name=None):
    assert not oneflow.eager_execution_enabled()
    return (
        oneflow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Unpack_")
        )
        .Op("unpack")
        .Input("in", [input])
        .Output("out")
        .Attr("unpack_num", unpack_num)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("pack")
def api_pack(
    input: oneflow_api.BlobDesc, pack_num: int, name: Optional[str] = None
) -> oneflow_api.BlobDesc:
    func = enable_if.unique([pack])
    return func(input, pack_num, name=name)


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def pack(input, pack_num, name=None):
    assert not oneflow.eager_execution_enabled()
    return (
        oneflow.user_op_builder(
            name if name is not None else id_util.UniqueStr("Pack_")
        )
        .Op("pack")
        .Input("in", [input])
        .Output("out")
        .Attr("pack_num", pack_num)
        .Build()
        .InferAndTryRun()
        .RemoteBlobList()[0]
    )


@oneflow_export("parallel_cast")
def api_parallel_cast(
    input: oneflow_api.BlobDesc,
    name: Optional[str] = None,
    distribute: Optional[oneflow_api.distribute.Distribute] = None,
    gradient_distribute: Optional[oneflow_api.distribute.Distribute] = None,
) -> oneflow_api.BlobDesc:
    func = enable_if.unique([parallel_cast])
    return func(
        input, name=name, distribute=distribute, gradient_distribute=gradient_distribute
    )


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def parallel_cast(input, name=None, distribute=None, gradient_distribute=None):
    if name is None:
        name = id_util.UniqueStr("ParallelCast_")

    def distribute_to_str(dist):
        dist_str = ""
        if dist is None:
            pass
        elif type(dist) is oneflow_api.distribute.SplitDistribute:
            dist_str = "S({})".format(dist.axis)
        elif type(dist) is oneflow_api.distribute.BroadcastDistribute:
            dist_str = "B"
        else:
            raise ValueError("unsupported distribute")
        return dist_str

    sbp_parallel = distribute_to_str(distribute)
    grad_sbp_parallel = distribute_to_str(gradient_distribute)
    op = (
        oneflow.user_op_builder(name)
        .Op("parallel_cast")
        .Input("in", [input])
        .Output("out")
        .Attr("sbp_parallel", sbp_parallel)
        .Attr("grad_sbp_parallel", grad_sbp_parallel)
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


@oneflow_export("hierarchical_parallel_cast")
def api_hierarchical_parallel_cast(
    input: oneflow_api.BlobDesc,
    parallel_distribution: Sequence[str],
    grad_mode: Optional[str] = None,
    grad_parallel_distribution: Sequence[str] = None,
    name: Optional[str] = None,
) -> oneflow_api.BlobDesc:
    func = enable_if.unique([hierarchical_parallel_cast])
    return func(
        input,
        parallel_distribution=parallel_distribution,
        grad_mode=grad_mode,
        grad_parallel_distribution=grad_parallel_distribution,
        name=name,
    )


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def hierarchical_parallel_cast(
    input, parallel_distribution, grad_mode, grad_parallel_distribution, name,
):
    if name is None:
        name = id_util.UniqueStr("HierarchicalParallelCast_")

    def distribute_to_str(dist):
        if dist is None:
            return ""
        elif type(dist) is str:
            return dist
        elif type(dist) is oneflow_api.distribute.SplitDistribute:
            return "S({})".format(dist.axis)
        elif type(dist) is oneflow_api.distribute.BroadcastDistribute:
            return "B"
        else:
            raise ValueError("unsupported distribute")

    op = (
        oneflow.user_op_builder(name)
        .Op("hierarchical_parallel_cast")
        .Input("in", [input])
        .Output("out")
        .Attr(
            "parallel_distribution", list(map(distribute_to_str, parallel_distribution))
        )
        .Attr("grad_mode", grad_mode or "restore")
        .Attr(
            "grad_parallel_distribution",
            list(map(distribute_to_str, grad_parallel_distribution))
            if grad_parallel_distribution
            else [],
        )
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()
