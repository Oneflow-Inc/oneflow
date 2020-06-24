from __future__ import absolute_import

import oneflow.core.job.sbp_parallel_pb2 as sbp_parallel_pb
import oneflow.core.job.mirrored_parallel_pb2 as mirrored_parallel_pb


class OpArgBlobAttribute(object):
    def __init__(self, batch_axis, blob_desc, logical_blob_name):
        self.batch_axis_ = batch_axis
        self.blob_desc_ = blob_desc
        self.shape_ = tuple(self.blob_desc_.body.shape.dim)
        self.logical_blob_name_ = logical_blob_name

    @property
    def shape(self):
        return self.shape_

    @property
    def dtype(self):
        return self.blob_desc_.body.data_type

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


class OpArgParallelAttribute(object):
    def __init__(self, parallel_desc_symbol, sbp_parallel, opt_mirrored_parallel):
        self.parallel_desc_symbol_ = parallel_desc_symbol
        self.sbp_parallel_ = sbp_parallel
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

    def __hash__(self):
        return self.hash_

    def __eq__(self, other):
        return (
            self.parallel_desc_symbol_ == other.parallel_desc_symbol_
            and self.opt_mirrored_parallel_ == other.opt_mirrored_parallel_
            and (
                self.opt_mirrored_parallel_.HasField("mirrored_parallel")
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
        if self.opt_mirrored_parallel_.HasField("mirrored_parallel"):
            sbp_hash = 0
        else:
            sbp_hash = hash(str(self.sbp_parallel_))
        return (
            hash(self.parallel_desc_symbol_)
            ^ hash(str(self.opt_mirrored_parallel_))
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
