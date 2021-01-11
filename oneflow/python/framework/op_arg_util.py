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

from google.protobuf import text_format
import oneflow.core.job.sbp_parallel_pb2 as sbp_parallel_pb
import oneflow.core.job.mirrored_parallel_pb2 as mirrored_parallel_pb
import oneflow.core.register.blob_desc_pb2 as blob_desc_pb
import oneflow.python.framework.balanced_splitter as balanced_splitter
from oneflow.python.framework.dtype import convert_proto_dtype_to_oneflow_dtype
import oneflow_api


class OpArgBlobAttribute(oneflow_api.OpArgBlobAttribute):
    def __init__(self, batch_axis, blob_desc, logical_blob_name):
        if not isinstance(batch_axis, oneflow_api.CfgMessage):
            batch_axis = oneflow_api.deprecated.MakeOptInt64ByString(str(batch_axis))

        if not isinstance(blob_desc, oneflow_api.CfgMessage):
            if str(blob_desc) is "":
                blob_desc = oneflow_api.oneflow.core.register.blob_desc.BlobDescProto()
            else:
                blob_desc = oneflow_api.deprecated.MakeBlobDescProtoByString(
                    str(blob_desc)
                )
        oneflow_api.OpArgBlobAttribute.__init__(
            self, batch_axis, blob_desc, logical_blob_name
        )

    @property
    def dtype(self):
        return convert_proto_dtype_to_oneflow_dtype(self.get_dtype())

    def GetPhysicalOpArgBlobAttr(self, split_axis, parallel_num, parallel_id):
        blob_desc = blob_desc_pb.BlobDescProto()
        text_format.Parse(str(self.blob_desc), blob_desc)
        physical_len = balanced_splitter.BalancedPartNums(
            self.shape[split_axis], parallel_num
        )[parallel_id]
        blob_desc.body.shape.dim[split_axis] = physical_len
        physical_blob_attr = OpArgBlobAttribute(
            self.batch_axis, blob_desc, self.logical_blob_name,
        )
        return physical_blob_attr

    def DumpToOpNodeSignature(self, bn_in_op, op_node_signature):
        blob_sig = op_node_signature.logical_blob_desc_signature.bn_in_op2blob_desc
        assert bn_in_op not in blob_sig
        text_format.Parse(str(self.blob_desc), blob_sig[bn_in_op])
        batch_axis_sig = op_node_signature.batch_axis_signature.bn_in_op2batch_axis
        assert bn_in_op not in batch_axis_sig
        text_format.Parse(str(self.batch_axis), batch_axis_sig[bn_in_op])


class OpArgParallelAttribute(oneflow_api.OpArgParallelAttribute):
    def __init__(self, parallel_desc_symbol, sbp_parallel, opt_mirrored_parallel):
        if not isinstance(sbp_parallel, oneflow_api.CfgMessage):
            sbp_parallel = oneflow_api.deprecated.MakeSbpParrallelByString(
                str(sbp_parallel)
            )
        if not isinstance(opt_mirrored_parallel, oneflow_api.CfgMessage):
            opt_mirrored_parallel = oneflow_api.deprecated.MakeOptMirroredParrallelByString(
                str(opt_mirrored_parallel)
            )
        oneflow_api.OpArgParallelAttribute.__init__(
            self, parallel_desc_symbol, sbp_parallel, opt_mirrored_parallel
        )

    def DumpToOpNodeSignature(self, bn_in_op, op_node_signature):
        sbp_sig = op_node_signature.sbp_signature.bn_in_op2sbp_parallel
        assert bn_in_op not in sbp_sig
        text_format.Parse(str(self.sbp_parallel), sbp_sig[bn_in_op])
        mirrored_sig = (
            op_node_signature.mirrored_signature.bn_in_op2opt_mirrored_parallel
        )
        assert bn_in_op not in mirrored_sig
        text_format.Parse(str(self.opt_mirrored_parallel), mirrored_sig[bn_in_op])
        parallel_sig = (
            op_node_signature.parallel_signature.bn_in_op2parallel_desc_symbol_id
        )
        assert bn_in_op not in parallel_sig
        parallel_sig[bn_in_op] = self.parallel_desc_symbol.symbol_id


def GetOpArgBlobAttribute(op_attribute, bn_in_op):
    if not op_attribute.HasField("batch_axis_signature"):
        return None
    if not op_attribute.HasField("logical_blob_desc_signature"):
        return None
    batch_axis_signature_map = op_attribute.batch_axis_signature.bn_in_op2batch_axis
    blob_desc_signature_map = (
        op_attribute.logical_blob_desc_signature.bn_in_op2blob_desc
    )
    arg_signature_map = op_attribute.arg_signature.bn_in_op2lbi
    lbi = arg_signature_map[bn_in_op]
    return OpArgBlobAttribute(
        batch_axis=batch_axis_signature_map[bn_in_op],
        blob_desc=blob_desc_signature_map[bn_in_op],
        logical_blob_name="%s/%s" % (lbi.op_name, lbi.blob_name),
    )


def GetOpArgParallelAttribute(parallel_desc_symbol, op_attribute, bn_in_op):
    sbp_signature_map = op_attribute.sbp_signature.bn_in_op2sbp_parallel
    mirrored_signature_map = (
        op_attribute.mirrored_signature.bn_in_op2opt_mirrored_parallel
    )
    return OpArgParallelAttribute(
        parallel_desc_symbol=parallel_desc_symbol,
        sbp_parallel=sbp_signature_map[bn_in_op],
        opt_mirrored_parallel=mirrored_signature_map[bn_in_op],
    )


def MakeMirroredOpArgParallelAttribute(parallel_desc_symbol):
    sbp_parallel = sbp_parallel_pb.SbpParallel()
    opt_mirrored_parallel = mirrored_parallel_pb.OptMirroredParallel()
    opt_mirrored_parallel.mirrored_parallel.SetInParent()
    return OpArgParallelAttribute(
        parallel_desc_symbol=parallel_desc_symbol,
        sbp_parallel=sbp_parallel,
        opt_mirrored_parallel=opt_mirrored_parallel,
    )


def MakeBroadcastOpArgParallelAttribute(parallel_desc_symbol):
    sbp_parallel = sbp_parallel_pb.SbpParallel()
    sbp_parallel.broadcast_parallel.SetInParent()
    opt_mirrored_parallel = mirrored_parallel_pb.OptMirroredParallel()
    return OpArgParallelAttribute(
        parallel_desc_symbol=parallel_desc_symbol,
        sbp_parallel=sbp_parallel,
        opt_mirrored_parallel=opt_mirrored_parallel,
    )
