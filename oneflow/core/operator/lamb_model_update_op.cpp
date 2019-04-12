#include "oneflow/core/operator/lamb_model_update_op.h"

namespace oneflow {

void LAMBModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  const auto& lamb_conf = op_conf().normal_mdupdt_conf().user_conf().lamb_conf();
  CHECK_GE(lamb_conf.beta1(), 0);
  CHECK_LT(lamb_conf.beta1(), 1);
  CHECK_GE(lamb_conf.beta2(), 0);
  CHECK_LT(lamb_conf.beta2(), 1);

  EnrollForwardModelBn("m");
  EnrollForwardModelBn("v");
  EnrollForwardModelBn("beta1_t");
  EnrollForwardModelBn("beta2_t");
  EnrollFwBufBn("fw_buf");
}

void LAMBModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ(model_blob_desc->data_type(), Global<JobDesc>::Get()->DefaultDataType());
  CHECK_EQ(model_blob_desc->has_data_id_field(), false);
  *GetBlobDesc4BnInOp("m") = *model_blob_desc;
  *GetBlobDesc4BnInOp("v") = *model_blob_desc;

  BlobDesc* beta1_t_blob = GetBlobDesc4BnInOp("beta1_t");
  BlobDesc* beta2_t_blob = GetBlobDesc4BnInOp("beta2_t");
  *beta1_t_blob = *model_blob_desc;
  *beta2_t_blob = *model_blob_desc;
  beta1_t_blob->set_data_type(DataType::kFloat);
  beta2_t_blob->set_data_type(DataType::kFloat);
  beta1_t_blob->mut_shape() = Shape({1});
  beta2_t_blob->mut_shape() = Shape({1});

  // fw_buf for gpu compute
  // fw_buf[0] for model_norm, fw_buf[1] for model_diff_norm
  *GetBlobDesc4BnInOp("fw_buf") = *model_blob_desc;
  GetBlobDesc4BnInOp("fw_buf")->mut_shape() = Shape({2});
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kLambConf, NormalModelUpdtOp, LAMBModelUpdateOp);

}  // namespace oneflow
