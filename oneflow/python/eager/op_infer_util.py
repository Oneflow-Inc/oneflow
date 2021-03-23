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
import oneflow.core.operator.op_node_signature_pb2 as op_node_signature_pb
import oneflow.python.framework.c_api_util as c_api_util
import oneflow_api.oneflow.core.operator.op_node_signature as op_node_signature_cfg
import oneflow
from google.protobuf import text_format


def Infer(op_conf, ibn2blob_object, scope_symbol_id=None):
    if scope_symbol_id is None:
        scope_symbol_id = oneflow.current_scope().symbol_id
    op_conf.scope_symbol_id = scope_symbol_id
    upstream_signature = MakeUpstreamSignature(ibn2blob_object)
    return c_api_util.InferOpConf(op_conf, upstream_signature)


def MakeUpstreamSignature(ibn2blob_object):
    upstream_signature_cfg = op_node_signature_cfg.OpNodeSignature()

    for ibn, blob_object in ibn2blob_object.items():
        blob_object.op_arg_blob_attr.DumpToOpNodeSignature(ibn, upstream_signature_cfg)
        blob_object.op_arg_parallel_attr.DumpToOpNodeSignature(
            ibn, upstream_signature_cfg
        )
    return text_format.Parse(
        str(upstream_signature_cfg), op_node_signature_pb.OpNodeSignature()
    )
