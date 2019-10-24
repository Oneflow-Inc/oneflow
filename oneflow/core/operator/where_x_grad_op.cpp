#include "oneflow/core/operator/where_x_grad_op.h"

namespace oneflow {

void WhereXGradOp::InitFromOpConf() {
  CHECK(op_conf().has_where_x_grad_conf());
  EnrollInputBn("condition", false);
  EnrollInputBn("out_diff");
  EnrollOutputBn("x_diff");
}

const PbMessage& WhereXGradOp::GetCustomizedConf() const { return op_conf().where_x_grad_conf(); }

Maybe<void> WhereXGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* condition_blob_desc = GetBlobDesc4BnInOp("condition");
  const BlobDesc* out_diff_blob_desc = GetBlobDesc4BnInOp("out_diff");
  CHECK_EQ_OR_RETURN(condition_blob_desc->shape().NumAxes(), out_diff_blob_desc->shape().NumAxes());
  FOR_RANGE(int64_t, i, 0, condition_blob_desc->shape().NumAxes()) {
    CHECK_EQ_OR_RETURN(condition_blob_desc->shape().At(i), out_diff_blob_desc->shape().At(i));
  }
  *GetBlobDesc4BnInOp("x_diff") = *GetBlobDesc4BnInOp("out_diff");
  return Maybe<void>::Ok();
}

Maybe<void> WhereXGradOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(
          JUST(LogicalBlobDesc4Ibn(input_bns().Get(0)))->shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kWhereXGradConf, WhereXGradOp);

}  // namespace oneflow
