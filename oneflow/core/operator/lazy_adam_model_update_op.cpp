#include "oneflow/core/operator/lazy_adam_model_update_op.h"

namespace oneflow {

void LazyAdamModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  const auto& lazy_adam_conf = op_conf().lazy_adam_model_update_conf().user_conf().lazy_adam_conf();
  CHECK_GE(lazy_adam_conf.beta1(), 0);
  CHECK_LT(lazy_adam_conf.beta1(), 1);
  CHECK_GE(lazy_adam_conf.beta2(), 0);
  CHECK_LT(lazy_adam_conf.beta2(), 1);

  EnrollInputBn("m", false)->set_is_mutable(true);
  EnrollInputBn("v", false)->set_is_mutable(true);
  EnrollInputBn("beta1_t", false)->set_is_mutable(true);
  EnrollInputBn("beta2_t", false)->set_is_mutable(true);
}

Maybe<void> LazyAdamModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ_OR_RETURN(model_blob_desc->data_type(), job_desc().DefaultDataType());
  CHECK_EQ_OR_RETURN(model_blob_desc->has_data_id_field(), false);
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("m") == *model_blob_desc);
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("v") == *model_blob_desc);

  CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("beta1_t")->shape(), Shape({1}));
  CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("beta2_t")->shape(), Shape({1}));
  return Maybe<void>::Ok();
}

const HashSet<std::string> LazyAdamModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{"beta1_t", "beta2_t"};
}

const PbMessage& LazyAdamModelUpdateOp::GetCustomizedConf() const {
  return op_conf().lazy_adam_model_update_conf();
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kLazyAdamConf, NormalModelUpdtOp,
               LazyAdamModelUpdateOp);

REGISTER_OP(OperatorConf::kLazyAdamModelUpdateConf, LazyAdamModelUpdateOp);

}  // namespace oneflow
