#include "oneflow/core/operator/relu_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ReluOp::InitFromOpConf() {
  CHECK(op_conf().has_relu_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in");
}

const PbMessage& ReluOp::GetCustomizedConf() const { return op_conf().relu_conf(); }

void ReluOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

void ReluOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(input_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kReluConf, ReluOp);

}  // namespace oneflow
