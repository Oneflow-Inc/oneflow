#include "oneflow/core/operator/switch_output_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/interface_op_util.h"

namespace oneflow {

void SwitchOutputOp::InitFromOpConf() {
  CHECK(op_conf().has_switch_output_conf());
  EnrollRepeatedInputBn("in");
  EnrollInputBn("in_index");
  EnrollOutputBn("out");
}

void SwitchOutputOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                    const ParallelContext* parallel_ctx) const {
  const BlobDesc& in_index_blob_desc = *GetBlobDesc4BnInOp("in_index");
  CHECK(in_index_blob_desc.shape() == Shape({1LL}));
  CHECK_EQ(in_index_blob_desc.data_type(), DataType::kInt32);
  const BlobDesc& first_in_blob_desc = *GetBlobDesc4BnInOp(GenRepeatedBn("in", 0));
  FOR_RANGE(int64_t, i, 0, op_conf().switch_output_conf().in_size()) {
    CHECK(*GetBlobDesc4BnInOp(GenRepeatedBn("in", i)) == first_in_blob_desc);
  }
  InterfaceOpUtil::InferOutBlobDesc(op_conf().switch_output_conf().blob_conf(),
                                    GetBlobDesc4BnInOp("out"), parallel_ctx);
  CHECK(*GetBlobDesc4BnInOp("out") == first_in_blob_desc);
}

const PbMessage& SwitchOutputOp::GetCustomizedConf() const {
  return op_conf().switch_output_conf();
}

void SwitchOutputOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  CHECK_EQ(*HasBatchDim4BnInOp("in_index"), false);
  bool first_in_has_batch_dim = *HasBatchDim4BnInOp(GenRepeatedBn("in", 0));
  FOR_RANGE(int64_t, i, 0, op_conf().switch_output_conf().in_size()) {
    CHECK_EQ(*HasBatchDim4BnInOp(GenRepeatedBn("in", i)), first_in_has_batch_dim);
  }
  InterfaceOpUtil::InferHasBatchDim(op_conf().switch_output_conf().blob_conf(),
                                    HasBatchDim4BnInOp("out"));
  CHECK(*HasBatchDim4BnInOp("out") == first_in_has_batch_dim);
}

void SwitchOutputOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  InterfaceOpUtil::GetOutputLikeOpSbpSignature(op_conf().switch_output_conf().blob_conf(),
                                               input_bns(), output_bns(),
                                               sbp_sig_list->mutable_sbp_signature()->Add());
}

REGISTER_OP(OperatorConf::kSwitchOutputConf, SwitchOutputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_MEM_BLOCK_NUM(OperatorConf::kSwitchOutputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kSwitchOutputConf);

}  // namespace oneflow
