#include "oneflow/core/operator/segment_sum_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SegmentSumGradOp::InitFromOpConf() {
  CHECK(op_conf().has_segment_sum_grad_conf());
  EnrollInputBn("out_diff", false);
  EnrollOutputBn("in_diff", false);
}

void SegmentSumGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                  const ParallelContext* parallel_ctx) const {
  const BlobDesc* out_diff_blob = GetBlobDesc4BnInOp("out_diff");
  const BlobDesc* segment_ids_blob = GetBlobDesc4BnInOp("segment_ids");
  // segment_ids must be 1D tensor
  CHECK_EQ(segment_ids_blob->shape().NumAxes(), 1);
  auto segment_ids_count = segment_ids_blob->shape().At(0);
  auto dims = out_diff_blob->shape().dim_vec();
  dims[0] = segment_ids_count;
  BlobDesc* in_diff_blob = GetBlobDesc4BnInOp("in_diff");
  in_diff_blob->set_data_type(out_diff_blob->data_type());
  in_diff_blob->mut_shape() = Shape(dims);
}

void SegmentSumGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("in_dff", 0)
      .Split("out_diff", 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn("out_idff").shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kSegmentSumGradConf, SegmentSumGradOp);

}  // namespace oneflow
