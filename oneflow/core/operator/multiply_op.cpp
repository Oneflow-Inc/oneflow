#include "oneflow/core/operator/multiply_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void MultiplyOp::InitFromOpConf() {
  CHECK(op_conf().has_multiply_conf());
  EnrollInputBn("in_0");
  EnrollInputBn("in_1");
  EnrollOutputBn("out")->set_mutable_inplace_ibn("in_0");
}

const PbMessage& MultiplyOp::GetCustomizedConf() const { return op_conf().multiply_conf(); }

Maybe<void> MultiplyOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* in_0_blob_desc = GetBlobDesc4BnInOp("in_0");
  BlobDesc* in_1_blob_desc = GetBlobDesc4BnInOp("in_1");
  CHECK_EQ_OR_RETURN(in_0_blob_desc->data_type(), GlobalJobDesc().DefaultDataType());
  CHECK_EQ_OR_RETURN(in_0_blob_desc->shape(), in_1_blob_desc->shape());
  // out
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  *out_blob_desc = *in_0_blob_desc;
  return Maybe<void>::Ok();
}

void MultiplyOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(output_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kMultiplyConf, MultiplyOp);

}  // namespace oneflow
