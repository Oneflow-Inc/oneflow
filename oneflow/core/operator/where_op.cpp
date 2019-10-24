#include "oneflow/core/operator/where_op.h"

namespace oneflow {

void WhereOp::InitFromOpConf() {
  CHECK(op_conf().has_where_conf());
  EnrollInputBn("condition", false);
  EnrollInputBn("x");
  EnrollInputBn("y");
  EnrollOutputBn("out");
}

const PbMessage& WhereOp::GetCustomizedConf() const { return op_conf().where_conf(); }

Maybe<void> WhereOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  const BlobDesc* condition_blob_desc = GetBlobDesc4BnInOp("condition");
  const BlobDesc* x_blob_desc = GetBlobDesc4BnInOp("x");
  const BlobDesc* y_blob_desc = GetBlobDesc4BnInOp("y");
  CHECK_EQ_OR_RETURN(condition_blob_desc->shape().NumAxes(), x_blob_desc->shape().NumAxes());
  CHECK_EQ_OR_RETURN(condition_blob_desc->shape().NumAxes(), y_blob_desc->shape().NumAxes());
  FOR_RANGE(int64_t, i, 0, condition_blob_desc->shape().NumAxes()) {
    CHECK_EQ_OR_RETURN(condition_blob_desc->shape().At(i), x_blob_desc->shape().At(i));
    CHECK_EQ_OR_RETURN(condition_blob_desc->shape().At(i), y_blob_desc->shape().At(i));
  }
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("x");
  return Maybe<void>::Ok();
}

Maybe<void> WhereOp::GetSbpSignatures(
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

REGISTER_OP(OperatorConf::kWhereConf, WhereOp);

}  // namespace oneflow
