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

import re

import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow.core.register.logical_blob_id_pb2 as logical_blob_id_util
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.compile_context as compile_context
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.input_blob_def as input_blob_util
import oneflow.python.framework.remote_blob as remote_blob_util
import oneflow.python.framework.hob as hob
import oneflow.python.lib.core.enable_if as enable_if
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.scope_util as scope_util
import oneflow.python.eager.vm_util as vm_util
import oneflow.python.eager.blob_register as blob_register_util
import oneflow_api.oneflow.core.job.placement as placement_cfg
import oneflow_api

blob_register = blob_register_util.GetDefaultBlobRegister()


def InputOpByArgBlobDef(blob_def):
    assert isinstance(blob_def, input_blob_util.ArgBlobDef)
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = blob_def.op_name
    op_conf.input_conf.out = blob_def.blob_name
    op_conf.input_conf.blob_conf.CopyFrom(blob_def.ToInterfaceBlobConf())
    blob_def.AddAndInferOp(op_conf)
    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = blob_def.op_name
    lbi.blob_name = blob_def.blob_name
    return remote_blob_util.RemoteBlob(lbi)


def ReturnRemoteBlob(remote_blob, allow_cpu_return_op=True):
    return enable_if.unique([LazyReturnRemoteBlob, EagerReturnRemoteBlob])(
        remote_blob, allow_cpu_return_op
    )


@enable_if.condition(hob.in_global_mode & ~hob.eager_execution_enabled)
def LazyReturnRemoteBlob(remote_blob, allow_cpu_return_op=True):
    assert isinstance(
        remote_blob, (oneflow_api.LazyMirroredBlob, oneflow_api.LazyConsistentBlob),
    )
    op_conf, lbi, scope = _GetReturnOpConfAndOutLbiAndScope(
        remote_blob, allow_cpu_return_op
    )
    compile_context.CurJobAddOp(op_conf, scope)
    return remote_blob_util.RemoteBlob(lbi)


@enable_if.condition(hob.in_global_mode & hob.eager_execution_enabled)
def EagerReturnRemoteBlob(remote_blob, allow_cpu_return_op=True):
    if not hob.is_trainable(None):
        return remote_blob
    op_conf, lbi, scope = _GetReturnOpConfAndOutLbiAndScope(
        remote_blob, allow_cpu_return_op
    )
    if remote_blob.blob_object.op_arg_parallel_attr.is_mirrored():
        add_and_infer = compile_context.CurJobAddMirroredOp
    else:
        add_and_infer = compile_context.CurJobAddConsistentOp
    op_attribute = add_and_infer(op_conf, scope)

    def BuildInstruction(builder):
        get_blob_scope = blob_register.BnInOp2BlobObjectScope
        with get_blob_scope(op_attribute) as bn_in_op2blob_object:
            builder.StatelessCall(
                op_attribute,
                remote_blob.blob_object.parallel_desc_symbol.parallel_conf,
                bn_in_op2blob_object=bn_in_op2blob_object,
            )

    vm_util.LogicalRun(BuildInstruction)
    return remote_blob_util.RemoteBlob(lbi)


def _GetReturnOpConfAndOutLbiAndScope(remote_blob, allow_cpu_return_op=True):
    op_conf = op_conf_util.OperatorConf()
    op_conf.name = id_util.UniqueStr("Return_")
    setattr(op_conf.return_conf, "in", remote_blob.unique_name)
    op_conf.return_conf.out = "out"
    if allow_cpu_return_op:
        op_conf.device_tag = "cpu"

    lbi = logical_blob_id_util.LogicalBlobId()
    lbi.op_name = op_conf.name
    lbi.blob_name = "out"

    parallel_conf = placement_cfg.ParallelConf()
    parallel_conf.CopyFrom(remote_blob.parallel_conf)

    def BuildScope(old_scope, builder):
        return builder.BuildScopeWithNewParallelConf(old_scope, parallel_conf)

    sess = session_ctx.GetDefaultSession()
    scope = scope_util.MakeScope(BuildScope)

    return op_conf, lbi, scope
