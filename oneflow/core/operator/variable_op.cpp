#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void VariableOp::InitFromOpConf() {
  CHECK(op_conf().has_variable_conf());
  if (op_conf().variable_conf().has_tick()) { EnrollInputBn("tick", false); }
  EnrollOutputBn("out", GlobalJobDesc().IsTrain())->set_is_mutable(true);
}

const PbMessage& VariableOp::GetCustomizedConf() const { return op_conf().variable_conf(); }

Maybe<void> VariableOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const VariableOpConf& variable_conf = op_conf().variable_conf();
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape(variable_conf.shape());
  out_blob_desc->set_data_type(variable_conf.has_data_type() ? variable_conf.data_type()
                                                             : GlobalJobDesc().DefaultDataType());
  if (parallel_ctx->policy() == kModelParallel) {
    int32_t model_split_axis = variable_conf.model_split_axis();
    CHECK_GE_OR_RETURN(model_split_axis, 0);
    CHECK_LT_OR_RETURN(model_split_axis, out_blob_desc->shape().NumAxes());
    int64_t split_dim_num = out_blob_desc->shape().At(model_split_axis);
    BalancedSplitter bs(split_dim_num, parallel_ctx->parallel_num());
    out_blob_desc->mut_shape().Set(model_split_axis, bs.At(parallel_ctx->parallel_id()).size());
  } else {
    CHECK_EQ_OR_RETURN(parallel_ctx->policy(), kDataParallel);
  }
  return Maybe<void>::Ok();
}

Maybe<void> VariableOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = false;
  return Maybe<void>::Ok();
}

void VariableOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  FOR_RANGE(int32_t, i, 0, op_conf().variable_conf().shape().dim_size()) {
    SbpSignatureBuilder()
        .Broadcast(input_bns())
        .Split(output_bns(), i)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  SbpSignatureBuilder()
      .Broadcast(input_bns())
      .Broadcast(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

Maybe<void> VariableOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  SbpSignature var_sbp_sig_conf(sbp_sig_conf);
  if (sbp_sig_conf.bn_in_op2sbp_parallel().empty()) {
    (*var_sbp_sig_conf.mutable_bn_in_op2sbp_parallel())["out"].mutable_broadcast_parallel();
  }
  return this->Operator::InferSbpSignature(sbp_signature, var_sbp_sig_conf, CalcOrderValue4SbpSig,
                                           SbpInferHint4Ibn, parallel_desc);
}

REGISTER_OP(OperatorConf::kVariableConf, VariableOp);
REGISTER_OP_SAME_OUTPUT_BLOB_MEM_BLOCK_NUM(OperatorConf::kVariableConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kVariableConf);

}  // namespace oneflow
