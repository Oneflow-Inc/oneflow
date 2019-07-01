#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void VariableOp::InitFromOpConf() {
  CHECK(op_conf().has_variable_conf());
  if (op_conf().variable_conf().has_tick()) { EnrollInputBn("tick", false); }
  bool has_diff =
      Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf();
  EnrollOutputBn("out", has_diff)->set_is_mutable(true);
  EnrollModelBn(op_conf().variable_conf().model_name());
}

const PbMessage& VariableOp::GetCustomizedConf() const { return op_conf().variable_conf(); }

void VariableOp::InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                                const ParallelContext* parallel_ctx) const {
  const VariableOpConf& variable_conf = op_conf().variable_conf();
  BlobDesc* model_blob_desc = GetBlobDesc4BnInOp(variable_conf.model_name());
  model_blob_desc->mut_shape() = Shape(variable_conf.shape());
  model_blob_desc->set_data_type(variable_conf.has_data_type()
                                     ? variable_conf.data_type()
                                     : Global<JobDesc>::Get()->DefaultDataType());
  if (parallel_ctx->policy() == kModelParallel) {
    int32_t model_split_axis = variable_conf.model_split_axis();
    CHECK_GE(model_split_axis, 0);
    CHECK_LT(model_split_axis, model_blob_desc->shape().NumAxes());
    int64_t split_dim_num = model_blob_desc->shape().At(model_split_axis);
    BalancedSplitter bs(split_dim_num, parallel_ctx->parallel_num());
    model_blob_desc->mut_shape().Set(model_split_axis, bs.At(parallel_ctx->parallel_id()).size());
  } else {
    CHECK_EQ(parallel_ctx->policy(), kDataParallel);
  }
  *GetBlobDesc4BnInOp("out") = *model_blob_desc;
}

void VariableOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* conf) const {
  conf->mutable_variable_conf()->set_is_fw_inplace(*is_fw_inplace_);
  conf->mutable_variable_conf()->set_is_bw_inplace(*is_bw_inplace_);
}

void VariableOp::InferHasBatchDim(
    std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const {
  *HasBatchDim4BnInOp("out") = false;
}

void VariableOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  FOR_RANGE(int32_t, i, 0, op_conf().variable_conf().shape().dim_size()) {
    SbpSignatureBuilder()
        .Split(input_bns(), 0)
        .Split(output_bns(), i)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  SbpSignatureBuilder()
      .Split(input_bns(), 0)
      .Broadcast(output_bns())
      .Build(sbp_sig_list->mutable_sbp_signature()->Add());
}

void VariableOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  SbpSignature var_sbp_sig_conf(sbp_sig_conf);
  if (sbp_sig_conf.bn_in_op2sbp_parallel().empty()) {
    (*var_sbp_sig_conf.mutable_bn_in_op2sbp_parallel())["out"].mutable_broadcast_parallel();
  }
  this->Operator::InferSbpSignature(sbp_signature, var_sbp_sig_conf, CalcOrderValue4SbpSig,
                                    SbpInferHint4Ibn, parallel_desc);
}

REGISTER_OP(OperatorConf::kVariableConf, VariableOp);

}  // namespace oneflow
