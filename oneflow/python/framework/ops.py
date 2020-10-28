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
from typing import Union, Optional


@oneflow_export("repeat")
def api_repeat(
    input: remote_blob_util.BlobDef, repeat_num: int, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
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
    one: remote_blob_util.BlobDef, max_acc_num: int, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
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
    input: remote_blob_util.BlobDef, unpack_num: int, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
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
    input: remote_blob_util.BlobDef, pack_num: int, name: Optional[str] = None
) -> remote_blob_util.BlobDef:
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
    input: remote_blob_util.BlobDef,
    name: Optional[str] = None,
    distribute: Optional[distribute_util.Distribute] = None,
    gradient_distribute: Optional[distribute_util.Distribute] = None,
) -> remote_blob_util.BlobDef:
    func = enable_if.unique([parallel_cast])
    return func(
        input, name=name, distribute=distribute, gradient_distribute=gradient_distribute
    )


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def parallel_cast(input, name=None, distribute=None, gradient_distribute=None):
    assert not oneflow.eager_execution_enabled()
    op_conf = op_conf_util.OperatorConf()
    setattr(
        op_conf,
        "name",
        name if name is not None else id_util.UniqueStr("ParallelCast_"),
    )
    op_conf.parallel_cast_conf.out = "out"
    setattr(op_conf.parallel_cast_conf, "in", input.unique_name)

    def to_split_axis(dist):
        split_axis = data_type_util.OptInt64()
        if type(dist) is distribute_util.SplitDistribute:
            split_axis.value = dist.axis
        elif type(dist) is distribute_util.BroadcastDistribute:
            split_axis.ClearField("value")
        else:
            raise NotImplementedError
        return split_axis

    if distribute is not None:
        op_conf.parallel_cast_conf.split_axis.CopyFrom(to_split_axis(distribute))
    if gradient_distribute is not None:
        op_conf.parallel_cast_conf.gradient_split_axis.CopyFrom(
            to_split_axis(gradient_distribute)
        )
    compile_context.CurJobAddOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"
    return remote_blob_util.RemoteBlob(lbi)
