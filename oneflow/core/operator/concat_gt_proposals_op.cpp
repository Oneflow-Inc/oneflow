#include "oneflow/core/operator/concat_gt_proposals_op.h"

namespace oneflow {

void ConcatGtProposalsOp::InitFromOpConf() {
  CHECK_EQ(device_type(), DeviceType::kCPU);
  CHECK(op_conf().has_concat_gt_proposals_conf());
  // input
  EnrollInputBn("in", false);
  EnrollInputBn("gt_boxes", false);
  // output
  EnrollOutputBn("out", false);
}

const PbMessage& ConcatGtProposalsOp::GetCustomizedConf() const {
  return op_conf().concat_gt_proposals_conf();
}

void ConcatGtProposalsOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  // in: in_proposals (R, 5) T
  const BlobDesc* in_proposals_blob_desc = GetBlobDesc4BnInOp("in");
  CHECK(in_proposals_blob_desc->has_dim0_valid_num_field());
  // in: gt_boxes (N, B, 4) T
  const BlobDesc* gt_boxes_blob_desc = GetBlobDesc4BnInOp("gt_boxes");
  CHECK_EQ(in_proposals_blob_desc->data_type(), gt_boxes_blob_desc->data_type());
  CHECK(gt_boxes_blob_desc->has_dim1_valid_num_field());

  const int64_t num_proposals =
      in_proposals_blob_desc->shape().At(0) + gt_boxes_blob_desc->shape().Count(0, 2);
  // out: out_proposals (R`, 5) T
  BlobDesc* out_proposals_blob_desc = GetBlobDesc4BnInOp("out");
  *out_proposals_blob_desc = *in_proposals_blob_desc;
  out_proposals_blob_desc->mut_shape().Set(0, num_proposals);
  out_proposals_blob_desc->mut_dim0_inner_shape() = Shape({1, num_proposals});
}

REGISTER_CPU_OP(OperatorConf::kConcatGtProposalsConf, ConcatGtProposalsOp);

}  // namespace oneflow
