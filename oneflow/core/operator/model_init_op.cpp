#include "oneflow/core/operator/model_init_op.h"

namespace oneflow {

void ModelInitOp::InitFromOpConf() {
  CHECK(op_conf().has_model_init_conf());
  if (op_conf().model_init_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", false);
}

const PbMessage& ModelInitOp::GetCustomizedConf() const { return op_conf().model_init_conf(); }

Maybe<void> ModelInitOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const VariableOpConf& original_variable_conf =
      op_conf().model_init_conf().original_variable_conf();
  BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("out");
  model_blob_desc->mut_shape() = Shape(original_variable_conf.shape());
  model_blob_desc->set_data_type(original_variable_conf.data_type());
  return Maybe<void>::Ok();
}

Maybe<void> ModelInitOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = false;
  return Maybe<void>::Ok();
}

void ModelInitOp::GetSbpSignatures(
    const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
    SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Split(output_bns(), 0)
      .MakeSplitSignatureListBuilder(LogicalBlobDesc4Ibn(output_bns().Get(0)).shape().NumAxes())
      .Build(sbp_sig_list);
}

REGISTER_OP(OperatorConf::kModelInitConf, ModelInitOp);

}  // namespace oneflow
