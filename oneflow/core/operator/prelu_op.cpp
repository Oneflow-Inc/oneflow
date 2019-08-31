#include "oneflow/core/operator/prelu_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void PReluOp::InitFromOpConf() {
  CHECK(op_conf().has_prelu_conf());
  const PReluOpConf& conf = op_conf().prelu_conf();
  StrFieldTolower("data_format");
  EnrollInputBn("in");
  if (conf.has_alpha()) {
    EnrollInputBn("alpha");
  } else {
    EnrollTmpBn("alpha");
  }
  EnrollOutputBn("out");
}

const PbMessage& PReluOp::GetCustomizedConf() const { return op_conf().prelu_conf(); }

Maybe<void> PReluOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  const PReluOpConf& conf = op_conf().prelu_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  int32_t alpha_size;
  if (conf.channel_shared()) {
    alpha_size = 1;
  } else {
    if (conf.data_format() == "channels_first") {
      alpha_size = in_blob_desc->shape().At(1);
    } else if (conf.data_format() == "channels_last") {
      alpha_size =
          in_blob_desc->shape().At(in_blob_desc->shape().NumAxes() - 1);
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
  const Shape alpha_shape({alpha_size});
  if (conf.has_alpha()) {
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("alpha")->shape(), alpha_shape);
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("alpha")->data_type(), in_blob_desc->data_type());
  } else {
    BlobDesc* alpha_blob_desc = GetBlobDesc4BnInOp("alpha");
    alpha_blob_desc->set_data_type(in_blob_desc->data_type());
    alpha_blob_desc->mut_shape() = alpha_shape;
  }
  return Maybe<void>::Ok();
}

Maybe<void> PReluOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = *HasBatchDim4BnInOp("in");
  return Maybe<void>::Ok();
}

void PReluOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(output_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kPreluConf, PReluOp);

}  // namespace oneflow
