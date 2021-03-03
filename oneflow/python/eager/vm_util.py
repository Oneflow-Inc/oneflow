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
from contextlib import contextmanager

import oneflow.core.eager.eager_symbol_pb2 as eager_symbol_pb
import oneflow.core.job.placement_pb2 as placement_pb
import oneflow.core.job.job_conf_pb2 as job_conf_pb
import oneflow.core.job.scope_pb2 as scope_pb
import oneflow.core.operator.op_conf_pb2 as op_conf_pb
import oneflow.core.operator.op_node_signature_pb2 as op_node_signature_pb
import oneflow.core.register.blob_desc_pb2 as blob_desc_pb
import oneflow.python.eager.blob_cache as blob_cache_util
import oneflow.python.eager.symbol as symbol_util
import oneflow.python.eager.symbol_storage as symbol_storage
import oneflow_api.oneflow.core.job.scope as scope_cfg
import oneflow.python.framework.balanced_splitter as balanced_splitter
import oneflow.python.framework.c_api_util as c_api_util
import oneflow.python.framework.id_util as id_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.python_callback as python_callback
import oneflow.python.framework.session_context as session_ctx
import oneflow.python.framework.python_interpreter_util as python_interpreter_util
import oneflow
import oneflow_api.oneflow.core.vm.instruction as instr_cfg
import oneflow_api.oneflow.core.job.placement as placement_cfg
import oneflow_api.oneflow.core.job.job_conf as job_conf_cfg
import oneflow_api.oneflow.core.operator.op_node_signature as op_node_signature_cfg
import oneflow_api.oneflow.core.eager.eager_symbol as eager_symbol_cfg
from google.protobuf import text_format
import oneflow_api


def PhysicalRun(build):
    return _Run(
        build,
        oneflow_api.vm.PhysicalIdGenerator(),
        oneflow_api.vm.RunPhysicalInstruction,
        _ReleasePhysicalObject,
    )


def LogicalRun(build):
    return _Run(
        build,
        oneflow_api.vm.LogicalIdGenerator(),
        oneflow_api.vm.RunLogicalInstruction,
        _ReleaseLogicalObject,
    )


def _Run(build, id_generator, run_api, release_object):
    instruction_list = session_ctx.GetDefaultSession().instruction_list
    eager_symbol_list = session_ctx.GetDefaultSession().eager_symbol_list
    assert isinstance(instruction_list, instr_cfg.InstructionListProto)
    assert isinstance(eager_symbol_list, eager_symbol_cfg.EagerSymbolList)
    build(
        oneflow_api.deprecated.InstructionsBuilder(
            id_generator, instruction_list, eager_symbol_list, release_object
        )
    )
    run_api(instruction_list, eager_symbol_list)
    instruction_list.clear_instruction()
    eager_symbol_list.clear_eager_symbol()


def _DefaultBlobObject4Ibn(ibn):
    raise NotImplementedError


@contextmanager
def CudaHostPinBlob(build, blob_object):
    build.CudaHostRegisterBlob(blob_object)
    try:
        yield
    finally:
        build.CudaHostUnregisterBlob(blob_object)


def _FindOrCreateDelegateBlobObject(
    builder, Fetch, x_blob_object, op_arg_parallel_attr
):
    if x_blob_object.op_arg_parallel_attr == op_arg_parallel_attr:
        return x_blob_object
    blob_cache = blob_cache_util.FindOrCreateBlobCache(x_blob_object)
    return blob_cache.GetCachedDelegateBlobObject(op_arg_parallel_attr, Fetch)


def _GetOpConfBlobNameAttr(pb_message, field):
    if hasattr(pb_message, field):
        return getattr(pb_message, field)
    m = re.search("_(\d+)$", field)
    assert m is not None
    blob_name = field[0 : -len(m.group(0))]
    index = int(m.group(0)[1:])
    assert hasattr(pb_message, blob_name), (pb_message, blob_name)
    repeated_field = getattr(pb_message, blob_name)
    assert index >= 0
    assert index < len(repeated_field)
    return repeated_field[index]


def _ReleaseLogicalObject(obj, is_shutting_down=python_interpreter_util.IsShuttingDown):
    if is_shutting_down():
        return
    LogicalRun(lambda builder: builder.DeleteObject(obj))


def _ReleasePhysicalObject(
    obj, is_shutting_down=python_interpreter_util.IsShuttingDown
):
    if is_shutting_down():
        return
    PhysicalRun(lambda builder: builder.DeleteObject(obj))
