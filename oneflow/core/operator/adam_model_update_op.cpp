#include "oneflow/core/operator/adam_model_update_op.h"

namespace oneflow {

void AdamModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  const auto& adam_conf = op_conf().adam_model_update_conf().user_conf().adam_conf();
  CHECK_GE(adam_conf.beta1(), 0);
  CHECK_LT(adam_conf.beta1(), 1);
  CHECK_GE(adam_conf.beta2(), 0);
  CHECK_LT(adam_conf.beta2(), 1);

  EnrollInputBn("m", false)->set_is_mutable(true);
  EnrollInputBn("v", false)->set_is_mutable(true);
  if (adam_conf.do_bias_correction()) {
    EnrollInputBn("beta1_t", false)->set_is_mutable(true);
    EnrollInputBn("beta2_t", false)->set_is_mutable(true);
  }
}

Maybe<void> AdamModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& adam_conf = op_conf().adam_model_update_conf().user_conf().adam_conf();
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ_OR_RETURN(model_blob_desc->data_type(), job_desc().DefaultDataType());
  CHECK_EQ_OR_RETURN(model_blob_desc->has_data_id_field(), false);
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("m") == *model_blob_desc);
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("v") == *model_blob_desc);

  if (adam_conf.do_bias_correction()) {
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("beta1_t")->shape(), Shape({1}));
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("beta2_t")->shape(), Shape({1}));
  }
  return Maybe<void>::Ok();
}

const HashSet<std::string> AdamModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{"beta1_t", "beta2_t"};
}

const PbMessage& AdamModelUpdateOp::GetCustomizedConf() const {
  return op_conf().adam_model_update_conf();
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kAdamConf, NormalModelUpdtOp, AdamModelUpdateOp);

REGISTER_OP(OperatorConf::kAdamModelUpdateConf, AdamModelUpdateOp);

}  // namespace oneflow
