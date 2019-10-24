#include "oneflow/core/operator/where_y_grad_op.h"

namespace oneflow {

void WhereYGradOp::InitFromOpConf() {
  CHECK(op_conf().has_where_y_grad_conf());
  EnrollInputBn("condition", false);
  EnrollInputBn("out_diff");
  EnrollOutputBn("y_diff");
}

const PbMessage& WhereYGradOp::GetCustomizedConf() const { return op_conf().where_y_grad_conf(); }

Maybe<void> WhereYGradOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("y_diff") = *GetBlobDesc4BnInOp("out_diff");
  return Maybe<void>::Ok();
}

Maybe<void> WhereYGradOp::GetSbpSignatures(
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

REGISTER_OP(OperatorConf::kWhereYGradConf, WhereYGradOp);

}  // namespace oneflow
