#include "oneflow/core/operator/adam_model_update_op.h"

namespace oneflow {

void AdamModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  const auto& adam_conf = op_conf().normal_mdupdt_conf().user_conf().adam_conf();
  CHECK_GE(adam_conf.beta1(), 0);
  CHECK_LT(adam_conf.beta1(), 1);
  CHECK_GE(adam_conf.beta2(), 0);
  CHECK_LT(adam_conf.beta2(), 1);

  EnrollDataTmpBn("m");
  EnrollDataTmpBn("v");
  if (adam_conf.do_bias_correction()) {
    EnrollForwardModelBn("beta1_t");
    EnrollForwardModelBn("beta2_t");
  }
}

void AdamModelUpdateOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& adam_conf = op_conf().normal_mdupdt_conf().user_conf().adam_conf();
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ(model_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  CHECK_EQ(model_blob_desc->has_data_id_field(), false);
  *GetBlobDesc4BnInOp("m") = *model_blob_desc;
  *GetBlobDesc4BnInOp("v") = *model_blob_desc;

  if (adam_conf.do_bias_correction()) {
    *GetBlobDesc4BnInOp("beta1_t") = *model_blob_desc;
    *GetBlobDesc4BnInOp("beta2_t") = *model_blob_desc;
    GetBlobDesc4BnInOp("beta1_t")->mut_shape() = Shape({1});
    GetBlobDesc4BnInOp("beta2_t")->mut_shape() = Shape({1});
  }
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kAdamConf, NormalModelUpdtOp, AdamModelUpdateOp);

}  // namespace oneflow
