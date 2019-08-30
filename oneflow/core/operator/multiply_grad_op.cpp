#include "oneflow/core/operator/multiply_grad_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void MultiplyGradOp::InitFromOpConf() {
  CHECK(op_conf().has_multiply_grad_conf());
  EnrollInputBn("out_diff", false);
  EnrollInputBn("in_0", false);
  EnrollInputBn("in_1", false);
  EnrollOutputBn("in_0_diff", false);
  EnrollOutputBn("in_1_diff", false);
  //EnrollOutputBn("out")->set_mutable_inplace_ibn("in_0");
}

void MultiplyGradOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("in_0_diff") = *HasBatchDim4BnInOp("out_diff"); 
  *HasBatchDim4BnInOp("in_1_diff") = *HasBatchDim4BnInOp("out_diff"); 
  
}

const PbMessage& MultiplyGradOp::GetCustomizedConf() const { return op_conf().multiply_grad_conf(); }

void MultiplyGradOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  const BlobDesc* out_diff_desc = GetBlobDesc4BnInOp("out_diff");
  const BlobDesc* in_0_desc = GetBlobDesc4BnInOp("in_0");
  const BlobDesc* in_1_desc = GetBlobDesc4BnInOp("in_1");

  CHECK_EQ(in_0_desc->shape(), in_1_desc->shape());
  CHECK_EQ(in_0_desc->shape(), out_diff_desc->shape());

  // desc of out blobs 
  BlobDesc* in_0_diff_desc = GetBlobDesc4BnInOp("in_0_diff");
  BlobDesc* in_1_diff_desc = GetBlobDesc4BnInOp("in_1_diff");

  *in_0_diff_desc = *in_1_desc;
  *in_1_diff_desc = *in_0_desc;
}

void MultiplyGradOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(output_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kMultiplyGradConf, MultiplyGradOp);

}  // namespace oneflow
