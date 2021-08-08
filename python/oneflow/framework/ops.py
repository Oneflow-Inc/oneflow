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
from typing import Optional, Sequence, Union

import oneflow
import oneflow._oneflow_internal
import oneflow.core.common.data_type_pb2 as data_type_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.framework.compile_context as compile_context
import oneflow.framework.distribute as distribute_util
import oneflow.framework.hob as hob
import oneflow.framework.id_util as id_util
import oneflow.framework.remote_blob as remote_blob_util
import oneflow.support.enable_if as enable_if


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


def api_unpack(
    input: oneflow._oneflow_internal.BlobDesc,
    unpack_num: int,
    name: Optional[str] = None,
) -> oneflow._oneflow_internal.BlobDesc:
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


def api_parallel_cast(
    input: oneflow._oneflow_internal.BlobDesc,
    name: Optional[str] = None,
    distribute: Optional[oneflow._oneflow_internal.distribute.Distribute] = None,
    gradient_distribute: Optional[
        oneflow._oneflow_internal.distribute.Distribute
    ] = None,
) -> oneflow._oneflow_internal.BlobDesc:
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
        elif type(dist) is oneflow._oneflow_internal.distribute.SplitDistribute:
            dist_str = "S({})".format(dist.axis)
        elif type(dist) is oneflow._oneflow_internal.distribute.BroadcastDistribute:
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

