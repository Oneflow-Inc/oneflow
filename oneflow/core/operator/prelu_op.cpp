#include "oneflow/core/operator/prelu_op.h"
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void PReluOp::InitFromOpConf() {
  CHECK(op_conf().has_prelu_conf());
  StrFieldTolower("data_format");
  EnrollInputBn("in");
  EnrollInputBn("alpha");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
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
      alpha_size = in_blob_desc->shape().At(in_blob_desc->shape().NumAxes() - 1);
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
  const Shape alpha_shape({alpha_size});
  CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("alpha")->shape(), alpha_shape);
  CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("alpha")->data_type(), in_blob_desc->data_type());
  return Maybe<void>::Ok();
}

Maybe<void> PReluOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  *BatchAxis4BnInOp("out") = *BatchAxis4BnInOp("in");
  return Maybe<void>::Ok();
}

Maybe<void> PReluOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(
          JUST(LogicalBlobDesc4Ibn(output_bns().Get(0)))->shape().NumAxes())
      .Build(sbp_sig_list);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kPreluConf, PReluOp);

}  // namespace oneflow
