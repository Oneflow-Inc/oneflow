#include "oneflow/core/operator/local_ring_boxing_op.h"

namespace oneflow {

void LocalRingBoxingOp::InitFromOpConf() {
  EnrollInputBn("in", false);
  EnrollInputBn("recv", false);
  EnrollOutputBn("out", false);
  EnrollOutputBn("send", false);
}

LogicalBlobId LocalRingBoxingOp::ibn2lbi(const std::string& input_bn) const {
  return GetCustomizedBoxingConf().lbi();
}

LogicalBlobId LocalRingBoxingOp::obn2lbi(const std::string& output_bn) const {
  return GetCustomizedBoxingConf().lbi();
}

const LocalRingBoxingConf& LocalRingBoxingOp::GetCustomizedBoxingConf() const {
  return GetMsgFromCustomizedConf<LocalRingBoxingConf>("local_ring_boxing_conf");
}

const PbMessage& LocalRingAllReduceOp::GetCustomizedConf() const {
  return op_conf().local_ring_all_reduce_conf();
}

void LocalRingAllReduceOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("send") = *GetBlobDesc4BnInOp("in");
}

REGISTER_OP(OperatorConf::kLocalRingAllReduceConf, LocalRingAllReduceOp);

}  // namespace oneflow
