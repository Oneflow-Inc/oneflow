#include "oneflow/core/operator/sparse_cross_entropy_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SparseCrossEntropyOp::InitFromOpConf() {
  CHECK(op_conf().has_sparse_cross_entropy_conf());
  EnrollInputBn("prediction");
  EnrollInputBn("label", false);
  EnrollOutputBn("out");
}

const PbMessage& SparseCrossEntropyOp::GetCustomizedConf() const {
  return op_conf().sparse_cross_entropy_conf();
}

Maybe<void> SparseCrossEntropyOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const BlobDesc* pred_blob_desc = GetBlobDesc4BnInOp("prediction");
  const BlobDesc* label_blob_desc = GetBlobDesc4BnInOp("label");
  CHECK_OR_RETURN(IsIntegralDataType(label_blob_desc->data_type()));
  CHECK_OR_RETURN(IsFloatingDataType(pred_blob_desc->data_type()));
  CHECK_EQ_OR_RETURN(pred_blob_desc->is_dynamic(), label_blob_desc->is_dynamic());
  CHECK_GE_OR_RETURN(pred_blob_desc->shape().NumAxes(), 2);
  const int64_t num_out_axes = pred_blob_desc->shape().NumAxes() - 1;
  CHECK_GE_OR_RETURN(label_blob_desc->shape().NumAxes(), num_out_axes);
  CHECK_EQ_OR_RETURN(label_blob_desc->shape().Count(num_out_axes), 1);
  FOR_RANGE(int64_t, i, 0, num_out_axes) {
    CHECK_EQ_OR_RETURN(pred_blob_desc->shape().At(i), label_blob_desc->shape().At(i));
  }
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *pred_blob_desc;
  out_blob_desc->mut_shape() = Shape(std::vector<int64_t>(
      pred_blob_desc->shape().dim_vec().cbegin(), pred_blob_desc->shape().dim_vec().cend() - 1));
  return Maybe<void>::Ok();
}

Maybe<void> SparseCrossEntropyOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kSparseCrossEntropyConf, SparseCrossEntropyOp);

}  // namespace oneflow
