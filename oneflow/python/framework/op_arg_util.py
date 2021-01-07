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


class OpArgBlobAttribute(object):
    def __init__(self, batch_axis, blob_desc, logical_blob_name):
        self.batch_axis_ = batch_axis
        self.blob_desc_ = blob_desc
        self.shape_ = tuple(self.blob_desc_.body.shape.dim)
        self.logical_blob_name_ = logical_blob_name

    def __eq__(self, rhs):
        return (
            self.shape == rhs.shape
            and self.dtype == rhs.dtype
            and self.batch_axis == rhs.batch_axis
            and self.is_tensor_list == rhs.is_tensor_list
            and self.is_dynamic == rhs.is_dynamic
            and self.logical_blob_name == rhs.logical_blob_name
        )

    @property
    def blob_desc(self):
        return self.blob_desc_

    @property
    def shape(self):
        return self.shape_

    @property
    def dtype(self):
        return convert_proto_dtype_to_oneflow_dtype(self.blob_desc_.body.data_type)

    @property
    def batch_axis(self):
        return self.batch_axis_

    @property
    def is_tensor_list(self):
        return self.blob_desc_.is_tensor_list

    @property
    def is_dynamic(self):
        return self.blob_desc_.is_dynamic

    @property
    def logical_blob_name(self):
        return self.logical_blob_name_

    def GetPhysicalOpArgBlobAttr(self, split_axis, parallel_num, parallel_id):
        blob_desc = blob_desc_pb.BlobDescProto()
        blob_desc.CopyFrom(self.blob_desc)
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
        blob_sig[bn_in_op].CopyFrom(self.blob_desc_)
        batch_axis_sig = op_node_signature.batch_axis_signature.bn_in_op2batch_axis
        assert bn_in_op not in batch_axis_sig
        batch_axis_sig[bn_in_op].CopyFrom(self.batch_axis_)

    def DumpToToInterfaceBlobConf(self, interface_blob_conf):
        interface_blob_conf.shape.dim.extend(self.shape)
        interface_blob_conf.data_type = self.blob_desc_.body.data_type
        interface_blob_conf.is_dynamic = self.is_dynamic
        interface_blob_conf.is_tensor_list = self.is_tensor_list
        interface_blob_conf.batch_axis.CopyFrom(self.batch_axis)


class OpArgParallelAttribute(object):
    def __init__(self, parallel_desc_symbol, sbp_parallel, opt_mirrored_parallel):
        self.parallel_desc_symbol_ = parallel_desc_symbol
        if not isinstance(sbp_parallel, oneflow_api.CfgMessage):
            sbp_parallel = oneflow_api.deprecated.MakeSbpParrallelByString(
                str(sbp_parallel)
            )
        self.sbp_parallel_ = sbp_parallel
        if not isinstance(opt_mirrored_parallel, oneflow_api.CfgMessage):
            opt_mirrored_parallel = oneflow_api.deprecated.MakeOptMirroredParrallelByString(
                str(opt_mirrored_parallel)
            )
        self.opt_mirrored_parallel_ = opt_mirrored_parallel
        self.hash_ = self._Hash()

    @property
    def parallel_desc_symbol(self):
        return self.parallel_desc_symbol_

    @property
    def sbp_parallel(self):
        return self.sbp_parallel_

    @property
    def opt_mirrored_parallel(self):
        return self.opt_mirrored_parallel_

    def is_mirrored(self):
        return self.opt_mirrored_parallel.has_mirrored_parallel()

    def Assign(self, other):
        self.__init__(
            other.parallel_desc_symbol, other.sbp_parallel, other.opt_mirrored_parallel
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

    def DumpToToInterfaceBlobConf(self, interface_blob_conf):
        if self.sbp_parallel.has_split_parallel():
            interface_blob_conf.split_axis.value = (
                self.sbp_parallel.split_parallel().axis()
            )
        else:
            interface_blob_conf.ClearField("split_axis")

    def __hash__(self):
        return self.hash_

    def __eq__(self, other):
        return (
            self.parallel_desc_symbol_ == other.parallel_desc_symbol_
            and self.opt_mirrored_parallel_ == other.opt_mirrored_parallel_
            and (
                self.opt_mirrored_parallel_.has_mirrored_parallel()
                or self.sbp_parallel_ == other.sbp_parallel_
            )
        )

    def __str__(self):
        return (
            "\nparallel_desc_symbol: %s\nsbp_parallel: %s\nopt_mirrored_parallel: %s\n"
            % (
                self.parallel_desc_symbol.parallel_conf,
                self.sbp_parallel,
                self.opt_mirrored_parallel,
            )
        )

    def _Hash(self):
        if self.opt_mirrored_parallel_.has_mirrored_parallel():
            sbp_hash = 0
        else:
            sbp_hash = hash(self.sbp_parallel_)
        return (
            hash(self.parallel_desc_symbol_)
            ^ hash(self.opt_mirrored_parallel_)
            ^ sbp_hash
        )


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
