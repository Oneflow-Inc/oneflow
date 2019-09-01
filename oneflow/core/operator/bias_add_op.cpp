#include "oneflow/core/operator/bias_add_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void BiasAddOp::InitFromOpConf() {
  CHECK(op_conf().has_bias_add_conf());
  EnrollInputBn("a");
  EnrollInputBn("b");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("a");
  EnrollConstBufBn("bias_multiplier");
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
  *GetBlobDesc4BnInOp("bias_multiplier") = *a_blob_desc;
  GetBlobDesc4BnInOp("bias_multiplier")->mut_shape() = Shape({a_blob_desc->shape().At(0), 1});
  return Maybe<void>::Ok();
}

void BiasAddOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split("a", 0)
      .Broadcast("b")
      .Split(output_bns(), 0)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  SbpSignatureBuilder()
      .Split("b", 0)
      .Broadcast("a")
      .Split(output_bns(), 1)
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kBiasAddConf, BiasAddOp);

}  // namespace oneflow
