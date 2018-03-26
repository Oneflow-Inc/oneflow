#include "oneflow/core/operator/normalization_op.h"

namespace oneflow {

bool NormalizationOp::HasScaleOrCenter() const {
  const auto& normalization_conf = op_conf().normalization_conf();
  return normalization_conf.center() || normalization_conf.scale();
}

void NormalizationOp::InitFromOpConf() {
  const auto& normalization_conf = op_conf().normalization_conf();
  CHECK_GT(normalization_conf.epsilon(), 0.f);
  CHECK_GE(normalization_conf.momentum(), 0);
  CHECK_LE(normalization_conf.momentum(), 1);
  EnrollInputBn("inputs");
  EnrollOutputBn("outputs");
  EnrollDataTmpBn("new_mean");
  EnrollDataTmpBn("new_variance");
  // norm_model
  EnrollOtherBn("moving_mean");
  EnrollOtherBn("moving_variance");

  if (normalization_conf.center()) { EnrollModelBn("beta"); }
  if (normalization_conf.scale()) { EnrollModelBn("gamma"); }
  if (HasScaleOrCenter()) { EnrollDataTmpBn("normalized_inputs"); }
  EnrollDataTmpBn("inv_var");
  EnrollModelTmpBn("tmp_storage_for_sum");
}

const PbMessage& NormalizationOp::GetCustomizedConf() const {
  return op_conf().normalization_conf();
}

void NormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& normalization_conf = op_conf().normalization_conf();
  const BlobDesc* inputs_blob_desc = GetBlobDesc4BnInOp("inputs");
  if (HasScaleOrCenter()) {
    *GetBlobDesc4BnInOp("normalized_inputs") = *inputs_blob_desc;
  }
  *GetBlobDesc4BnInOp("outputs") = *inputs_blob_desc;
  BlobDesc blob_desc(Shape({1}), Global<JobDesc>::Get()->DefaultDataType(),
                     false, false, 1);
  std::list<std::string> scalar_blob_names = {"moving_mean", "moving_variance",
                                              "inv_var"};
  std::list<std::string> bns_needless_in_predict = {"new_mean", "new_variance"};
  if (normalization_conf.center()) { scalar_blob_names.push_back("beta"); }
  if (normalization_conf.scale()) { scalar_blob_names.push_back("gamma"); }
  if (Global<JobDesc>::Get()->IsTrain()) {
    for (const std::string& bn : bns_needless_in_predict) {
      scalar_blob_names.push_back(bn);
    }
  }
  for (const auto& bn_in_op : scalar_blob_names) {
    GetBlobDesc4BnInOp(bn_in_op)->mut_shape() = Shape({1});
  }
  int64_t tmp_storage_size =
      std::sqrt(GetBlobDesc4BnInOp("inputs")->shape().elem_cnt());
  GetBlobDesc4BnInOp("tmp_storage_for_sum")->mut_shape() =
      Shape({tmp_storage_size + 1});
}

void NormalizationOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext*, KernelConf* kernel_conf) const {
  int64_t inv_elem_cnt = GetBlobDesc4BnInOp("inputs")->shape().elem_cnt();
  kernel_conf->mutable_normalization_conf()->set_inv_inputs_elem_cnt(
      1.0 / inv_elem_cnt);
}

REGISTER_OP(OperatorConf::kNormalizationConf, NormalizationOp);

}  // namespace oneflow
