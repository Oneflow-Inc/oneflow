#include "oneflow/core/operator/cast_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void CastOp::InitFromOpConf() {
  CHECK(op_conf().has_cast_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

const PbMessage& CastOp::GetCustomizedConf() const { return op_conf().cast_conf(); }

Maybe<void> CastOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  *out_blob_desc = *in_blob_desc;
  out_blob_desc->set_data_type(op_conf().cast_conf().data_type());
  return Maybe<void>::Ok();
}

Maybe<void> CastOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(JUST(LogicalBlobDesc4Ibn("in"))->shape().NumAxes())
      .Build(sbp_sig_list);
  SbpSignatureBuilder()
      .PartialSum(input_bns())
      .PartialSum(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kCastConf, CastOp);

}  // namespace oneflow
