#include "oneflow/core/operator/adam_model_update_op.h"

namespace oneflow {

void AdamModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  if (Global<JobDesc>::Get()->IsTrain()) {
    const auto& adam_conf = op_conf().normal_mdupdt_conf().user_conf().adam_conf();
    CHECK_GE(adam_conf.beta1(), 0);
    CHECK_LT(adam_conf.beta1(), 1);
    CHECK_GE(adam_conf.beta2(), 0);
    CHECK_LT(adam_conf.beta2(), 1);
    EnrollForwardModelBn("m");
    EnrollForwardModelBn("v");
    if (adam_conf.do_bias_correction()) {
      EnrollForwardModelBn("beta1_t");
      EnrollForwardModelBn("beta2_t");
    }
  } else if (Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
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
  } else {
    UNIMPLEMENTED();
  }
}

void AdamModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  if (Global<JobDesc>::Get()->IsTrain()) {
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
  } else if (Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    const auto& adam_conf = op_conf().adam_model_update_conf().user_conf().adam_conf();
    const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
    CHECK_EQ(model_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
    CHECK_EQ(model_blob_desc->has_data_id_field(), false);
    CHECK(*GetBlobDesc4BnInOp("m") == *model_blob_desc);
    CHECK(*GetBlobDesc4BnInOp("v") == *model_blob_desc);

    if (adam_conf.do_bias_correction()) {
      CHECK_EQ(GetBlobDesc4BnInOp("beta1_t")->shape(), Shape({1}));
      CHECK_EQ(GetBlobDesc4BnInOp("beta2_t")->shape(), Shape({1}));
    }
  } else {
    UNIMPLEMENTED();
  }
}

const PbMessage& AdamModelUpdateOp::GetCustomizedConf() const {
  if (Global<JobDesc>::Get()->IsTrain()) {
    return op_conf().normal_mdupdt_conf();
  } else if (Global<JobDesc>::Get()->other_conf().predict_conf().has_tmp_split_fw_bw_train_conf()) {
    return op_conf().adam_model_update_conf();
  } else {
    UNIMPLEMENTED();
  }
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kAdamConf, NormalModelUpdtOp, AdamModelUpdateOp);

REGISTER_OP(OperatorConf::kAdamModelUpdateConf, AdamModelUpdateOp);

}  // namespace oneflow
