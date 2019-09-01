#include "oneflow/core/operator/tanh_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void TanHOp::InitFromOpConf() {
  CHECK(op_conf().has_tanh_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
}

const PbMessage& TanHOp::GetCustomizedConf() const { return op_conf().tanh_conf(); }

Maybe<void> TanHOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                   const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

void TanHOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn("in").shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kTanhConf, TanHOp);

}  // namespace oneflow
