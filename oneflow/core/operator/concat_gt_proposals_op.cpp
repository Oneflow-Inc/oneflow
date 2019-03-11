#include "oneflow/core/operator/concat_gt_proposals_op.h"

namespace oneflow {

void ConcatGtProposalsOp::InitFromOpConf() {
  CHECK_EQ(device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_proposal_conf());
  // input
  EnrollInputBn("in", false);
  EnrollInputBn("gt_boxes", false);
  // output
  EnrollOutputBn("out", false);

  // EnrollDataTmpBn("anchors");
}

const PbMessage& ConcatGtProposalsOp::GetCustomizedConf() const {
  return op_conf().proposal_conf();
}

void ConcatGtProposalsOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {}

REGISTER_CPU_OP(OperatorConf::kConcatGtProposalsConf, ConcatGtProposalsOp);

}  // namespace oneflow
