#include "oneflow/core/operator/output_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void OutputOp::InitFromOpConf() {
  CHECK(op_conf().has_output_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out");
}

void OutputOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                              const ParallelContext* parallel_ctx) const {
  *GetBlobDesc4BnInOp("out") = *GetBlobDesc4BnInOp("in");
}

const PbMessage& OutputOp::GetCustomizedConf() const { return op_conf().output_conf(); }

void OutputOp::InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = *HasBatchDim4BnInOp("in");
}

void OutputOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  int64_t num_axes = LogicalBlobDesc4Ibn(input_bns().Get(0)).shape().NumAxes();
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(num_axes)
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kOutputConf, OutputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_MEM_BLOCK_NUM(OperatorConf::kOutputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kOutputConf);
REGISTER_OUTPUT_INTERFACE_OP(OperatorConf::kOutputConf);

}  // namespace oneflow
