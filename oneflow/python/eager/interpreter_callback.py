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

import oneflow.python.eager.gradient_util as gradient_util
import oneflow.python.eager.op_executor as op_executor
import oneflow.core.operator.op_attribute_pb2 as op_attribute_pb
import oneflow.core.job.job_conf_pb2 as job_conf_pb
import oneflow.core.job.scope_pb2 as scope_pb
import oneflow.core.job.placement_pb2 as placement_pb
from google.protobuf import text_format
import oneflow.python.framework.scope_util as scope_util
import oneflow.python.eager.symbol_storage as symbol_storage
import oneflow_api


def MakeScopeSymbol(job_conf, parallel_conf, is_mirrored):
    parallel_hierarchy = None
    if parallel_conf.has_hierarchy():
        parallel_hierarchy = oneflow_api.Size(tuple(parallel_conf.hierarchy().dim()))
    return scope_util.MakeInitialScope(
        job_conf,
        parallel_conf.device_tag(),
        list(parallel_conf.device_name()),
        parallel_hierarchy,
        is_mirrored,
    ).symbol_id


def MakeParallelDescSymbol(parallel_conf):
    symbol_id = None

    def BuildInstruction(builder):
        nonlocal symbol_id
        symbol_id = builder.GetParallelDescSymbol(parallel_conf).symbol_id

    oneflow_api.deprecated.LogicalRun(BuildInstruction)
    return symbol_id


def MirroredCast(op_attribute_str, parallel_conf):
    op_attribute = text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())
    blob_register = oneflow_api.GetDefaultBlobRegister()
    is_cast_to_mirrored = op_attribute.op_conf.HasField("cast_to_mirrored_conf")
    is_cast_from_mirrored = op_attribute.op_conf.HasField("cast_from_mirrored_conf")
    assert is_cast_to_mirrored or is_cast_from_mirrored
    _MirroredCastAndAddOutputBlobReleaser(op_attribute, blob_register)
    bw_blob_register = gradient_util.GetDefaultBackwardBlobRegister()
    gradient_util.TrySetBackwardUsedBlobObject(
        op_attribute, blob_register, bw_blob_register
    )


def InterpretCompletedOp(op_attribute_str, parallel_conf):
    op_attribute = text_format.Parse(op_attribute_str, op_attribute_pb.OpAttribute())
    blob_register = gradient_util.GetDefaultBackwardBlobRegister()
    _InterpretCompletedOp(op_attribute, parallel_conf, blob_register)
    gradient_util.ReleaseUnusedBlobObject(op_attribute, blob_register)


def _InterpretCompletedOp(op_attribute, parallel_conf, blob_register):
    return op_executor.Interpret(op_attribute, parallel_conf, blob_register)


def _MirroredCastAndAddOutputBlobReleaser(op_attribute, blob_register):
    op_executor.MirroredCast(op_attribute, blob_register)
    _AddOutputBlobObjectReleaser4InputBlobObject(op_attribute, blob_register)


def _AddOutputBlobObjectReleaser4InputBlobObject(op_attribute, blob_register):
    in_lbi = op_attribute.arg_signature.bn_in_op2lbi["in"]
    in_lbn = "%s/%s" % (in_lbi.op_name, in_lbi.blob_name)
    in_blob_object = blob_register.GetObject4BlobName(in_lbn)
    release = _MakeReleaser4MirroredCastBlobObject(op_attribute, blob_register)
    in_blob_object.add_releaser(release)


def _MakeReleaser4MirroredCastBlobObject(op_attribute, blob_register):
    def ReleaseMirroredBlobObject(obj):
        for obn in op_attribute.output_bns:
            lbi = op_attribute.arg_signature.bn_in_op2lbi[obn]
            lbn = "%s/%s" % (lbi.op_name, lbi.blob_name)
            blob_object = blob_register.GetObject4BlobName(lbn)
            blob_register.ClearObject4BlobName(lbn)

    return ReleaseMirroredBlobObject
