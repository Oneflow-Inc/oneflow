#include "oneflow/core/operator/variable_op.h"

namespace oneflow {

void VariableOp::InitFromOpConf() {
  CHECK(op_conf().has_variable_conf());
  EnrollInputBn("tick", false);
  EnrollOutputBn("out", Global<JobDesc>::Get()->IsTrain() && op_conf().trainable());
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
    CHECK_EQ(split_dim_num % parallel_ctx->parallel_num(), 0);
    model_blob_desc->mut_shape().Set(model_split_axis,
                                     split_dim_num / parallel_ctx->parallel_num());
  } else {
    CHECK_EQ(parallel_ctx->policy(), kDataParallel);
  }
  *GetBlobDesc4BnInOp("out") = *model_blob_desc;
}

void VariableOp::InferOutputBlobParallelDesc(
    std::function<BlobParallelDesc*(const std::string&)> BlobParallelDesc4BnInOp,
    const ParallelContext* parallel_context) const {
  BlobModelParallel* blob_model_parallel = BlobParallelDesc4BnInOp("out")->mut_model_parallel();
  if (parallel_context->policy() == kDataParallel) {
    blob_model_parallel->set_clone_num(parallel_context->parallel_num());
    blob_model_parallel->set_model_split_num(1);
  } else if (parallel_context->policy() == kModelParallel) {
    blob_model_parallel->set_clone_num(1);
    blob_model_parallel->set_model_split_num(parallel_context->parallel_num());
  } else {
    UNIMPLEMENTED();
  }
}

void VariableOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
    KernelConf* conf) const {
  conf->mutable_variable_conf()->set_is_fw_inplace(*is_fw_inplace_);
  conf->mutable_variable_conf()->set_is_bw_inplace(*is_bw_inplace_);
}

REGISTER_OP(OperatorConf::kVariableConf, VariableOp);

}  // namespace oneflow
