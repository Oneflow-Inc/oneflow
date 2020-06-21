from __future__ import absolute_import

import oneflow.core.job.sbp_parallel_pb2 as sbp_parallel_pb
import oneflow.core.job.mirrored_parallel_pb2 as mirrored_parallel_pb


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
