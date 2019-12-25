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

Maybe<void> SwitchOutputOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc& in_index_blob_desc = *GetBlobDesc4BnInOp("in_index");
  CHECK_OR_RETURN(in_index_blob_desc.shape() == Shape({1LL}));
  CHECK_EQ_OR_RETURN(in_index_blob_desc.data_type(), DataType::kInt32);
  const BlobDesc& first_in_blob_desc = *GetBlobDesc4BnInOp(GenRepeatedBn("in", 0));
  FOR_RANGE(int64_t, i, 0, op_conf().switch_output_conf().in_size()) {
    CHECK_OR_RETURN(*GetBlobDesc4BnInOp(GenRepeatedBn("in", i)) == first_in_blob_desc);
  }
  *GetBlobDesc4BnInOp("out") = first_in_blob_desc;
  if (first_in_blob_desc.is_dynamic() == false) {
    InterfaceOpUtil::InferOutBlobDesc(op_conf().switch_output_conf().blob_conf(),
                                      GetBlobDesc4BnInOp("out"), parallel_ctx);
    CHECK_OR_RETURN(*GetBlobDesc4BnInOp("out") == first_in_blob_desc);
  }
  return Maybe<void>::Ok();
}

const PbMessage& SwitchOutputOp::GetCustomizedConf() const {
  return op_conf().switch_output_conf();
}

Maybe<void> SwitchOutputOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  CHECK_EQ(BatchAxis4BnInOp("in_index")->has_value(), false);
  const OptInt64& first_in_batch_axis = *BatchAxis4BnInOp(GenRepeatedBn("in", 0));
  FOR_RANGE(int64_t, i, 0, op_conf().switch_output_conf().in_size()) {
    CHECK_OR_RETURN(*BatchAxis4BnInOp(GenRepeatedBn("in", i)) == first_in_batch_axis);
  }
  InterfaceOpUtil::InferBatchAxis(op_conf().switch_output_conf().blob_conf(),
                                  BatchAxis4BnInOp("out"));
  CHECK_OR_RETURN(*BatchAxis4BnInOp("out") == first_in_batch_axis);
  return Maybe<void>::Ok();
}

Maybe<void> SwitchOutputOp::GetSbpSignatures(
    const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  InterfaceOpUtil::GetOutputLikeOpSbpSignature(op_conf().switch_output_conf().blob_conf(),
                                               input_bns(), output_bns(),
                                               sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kSwitchOutputConf, SwitchOutputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kSwitchOutputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kSwitchOutputConf);

}  // namespace oneflow
