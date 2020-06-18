from __future__ import absolute_import

import oneflow.core.job.sbp_parallel_pb2 as sbp_parallel_pb
import oneflow.core.job.mirrored_parallel_pb2 as mirrored_parallel_pb


class OpArgAttribute(object):
    def __init__(self, parallel_desc_symbol, sbp_parallel, opt_mirrored_parallel):
        self.parallel_desc_symbol_ = parallel_desc_symbol
        self.sbp_parallel_ = sbp_parallel
        self.opt_mirrored_parallel_ = opt_mirrored_parallel

    @property
    def parallel_desc_symbol(self):
        return self.parallel_desc_symbol_

    @property
    def sbp_parallel(self):
        return self.sbp_parallel_

    @property
    def opt_mirrored_parallel(self):
        return self.opt_mirrored_parallel_


def GetOpArgAttribute(parallel_desc_symbol, op_attribute, bn_in_op):
    sbp_signature_map = op_attribute.sbp_signature.bn_in_op2sbp_parallel
    mirrored_signature_map = (
        op_attribute.mirrored_signature.bn_in_op2opt_mirrored_parallel
    )
    return OpArgAttribute(
        parallel_desc_symbol=parallel_desc_symbol,
        sbp_parallel=sbp_signature_map[bn_in_op],
        opt_mirrored_parallel=mirrored_signature_map[bn_in_op],
    )


def MakeMirroredOpArgAttribute(parallel_desc_symbol):
    sbp_parallel = sbp_parallel_pb.SbpParallel()
    opt_mirrored_parallel = mirrored_parallel_pb.OptMirroredParallel()
    opt_mirrored_parallel.mirrored_parallel.SetInParent()
    return OpArgAttribute(
        parallel_desc_symbol=parallel_desc_symbol,
        sbp_parallel=sbp_parallel,
        opt_mirrored_parallel=opt_mirrored_parallel,
    )


def MakeBroadcastOpArgAttribute(parallel_desc_symbol):
    sbp_parallel = sbp_parallel_pb.SbpParallel()
    sbp_parallel.broadcast_parallel.SetInParent()
    opt_mirrored_parallel = mirrored_parallel_pb.OptMirroredParallel()
    return OpArgAttribute(
        parallel_desc_symbol=parallel_desc_symbol,
        sbp_parallel=sbp_parallel,
        opt_mirrored_parallel=opt_mirrored_parallel,
    )
