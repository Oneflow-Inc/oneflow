#include "oneflow/core/operator/relu_grad_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ReluGradOp::InitFromOpConf() {
  CHECK(op_conf().has_relu_grad_conf());
  EnrollInputBn("y");
  EnrollInputBn("dy");
  EnrollOutputBn("dx")->set_mutable_inplace_ibn("dy");
}

const PbMessage& ReluGradOp::GetCustomizedConf() const { return op_conf().relu_grad_conf(); }

void ReluGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("dx") = *GetBlobDesc4BnInOp("y");
}

void ReluGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(input_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kReluGradConf, ReluGradOp);

}  // namespace oneflow
