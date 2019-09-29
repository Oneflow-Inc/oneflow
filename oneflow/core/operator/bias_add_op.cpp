#include "oneflow/core/operator/bias_add_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void BiasAddOp::InitFromOpConf() {
  CHECK(op_conf().has_bias_add_conf());
  EnrollInputBn("a");
  EnrollInputBn("b");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("a");
}

const PbMessage& BiasAddOp::GetCustomizedConf() const { return op_conf().bias_add_conf(); }

Maybe<void> BiasAddOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* a_blob_desc = GetBlobDesc4BnInOp("a");
  const BlobDesc* b_blob_desc = GetBlobDesc4BnInOp("b");
  const int32_t bias_add_axis = op_conf().bias_add_conf().axis();
  CHECK_EQ_OR_RETURN(b_blob_desc->shape().NumAxes(), 1);
  CHECK_GE_OR_RETURN(bias_add_axis, 0);
  CHECK_LT_OR_RETURN(bias_add_axis, a_blob_desc->shape().NumAxes());
  CHECK_EQ_OR_RETURN(a_blob_desc->shape().At(bias_add_axis), b_blob_desc->shape().At(0));
  *GetBlobDesc4BnInOp("out") = *a_blob_desc;
  return Maybe<void>::Ok();
}

Maybe<void> BiasAddOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int64_t axis = op_conf().bias_add_conf().axis();
  for (int64_t i = 0; i < JUST(LogicalBlobDesc4Ibn("a"))->shape().NumAxes(); ++i) {
    if (i == axis) { continue; }
    SbpSignatureBuilder().Split("a", i).Broadcast("b").Split("out", i).Build(
        sbp_sig_list->mutable_sbp_signature()->Add());
  }
  SbpSignatureBuilder()
      .Split("b", 0)
      .Split("a", axis)
      .Split("out", axis)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kBiasAddConf, BiasAddOp);

}  // namespace oneflow
