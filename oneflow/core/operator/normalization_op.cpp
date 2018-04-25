#include "oneflow/core/operator/normalization_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

struct NormalizationOpCtx : public OpContext {
  bool use_cudnn;
};

void NormalizationOp::InitFromOpConf() {
  const auto& conf = op_conf().normalization_conf();
  CHECK_GT(conf.epsilon(), 0.f);
  CHECK_GE(conf.momentum(), 0);
  CHECK_LE(conf.momentum(), 1);
  EnrollInputBn("in");
  EnrollOutputBn("out");
  EnrollDataTmpBn("new_mean");
  EnrollDataTmpBn("new_variance");
  EnrollForwardModelBn("moving_mean");
  EnrollForwardModelBn("moving_variance");
  if (conf.center()) {
    EnrollModelBn("beta");
  } else {
    EnrollDataTmpBn("beta_diff");
  }
  if (conf.scale()) {
    EnrollModelBn("gamma");
  } else {
    EnrollDataTmpBn("gamma_diff");
  }
  EnrollDataTmpBn("normalized_in");
  EnrollDataTmpBn("inv_var");
  EnrollDataTmpBn("before_by_axis_matrix");
  EnrollModelTmpBn("before_axis_sum_multiplier");
  EnrollModelTmpBn("after_axis_sum_multiplier");
  EnrollDataTmpBn("cache_mean_for_cudnn_bw");
  EnrollDataTmpBn("cache_inv_variance_for_cudnn_bw");
}

const PbMessage& NormalizationOp::GetCustomizedConf() const {
  return op_conf().normalization_conf();
}

bool NormalizationOp::NeedOutWhenBackward() const {
  ActivationType activation =
      static_cast<ActivationType>(GetEnumFromCustomizedConf("activation"));
  return activation != ActivationType::kNone;
}

void NormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, DeviceType device_type,
    std::function<void(OpContext*)> EnrollOpCtx) const {
  const auto& conf = op_conf().normalization_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const DataType in_data_type = in_blob_desc->data_type();
  CHECK_EQ(in_data_type, Global<JobDesc>::Get()->DefaultDataType());
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
  auto* op_ctx = new NormalizationOpCtx();
  EnrollOpCtx(op_ctx);
#ifdef WITH_CUDA
  int32_t in_dims = in_blob_desc->shape().NumAxes();
  if (device_type == DeviceType::kGPU && CUDNN_VERSION >= 5000
      && in_data_type == DataType::kFloat && in_dims >= 4 && in_dims <= 5
      && (conf.axis() == 1 || conf.axis() == in_dims - 1)) {
    op_ctx->use_cudnn = true;
    InferBlobDescsForCudnn(GetBlobDesc4BnInOp);
    return;
  }
#endif
  op_ctx->use_cudnn = false;
  InferParamBlobDescs(GetBlobDesc4BnInOp, conf,
                      in_blob_desc->shape().At(conf.axis()), in_data_type,
                      false);
  *GetBlobDesc4BnInOp("normalized_in") = *in_blob_desc;
  int64_t num_before_axis_dim =
      in_blob_desc->shape().CountBeforeAxis(conf.axis());
  int64_t num_axis_dim = in_blob_desc->shape().At(conf.axis());
  int64_t num_after_axis_dim =
      in_blob_desc->shape().CountAfterAxis(conf.axis());
  BlobDesc* before_by_axis_blob_desc =
      GetBlobDesc4BnInOp("before_by_axis_matrix");
  before_by_axis_blob_desc->mut_shape() =
      Shape({num_before_axis_dim, num_axis_dim});
  before_by_axis_blob_desc->set_data_type(in_data_type);
  BlobDesc* after_axis_sum_multiplier =
      GetBlobDesc4BnInOp("after_axis_sum_multiplier");
  after_axis_sum_multiplier->mut_shape() = Shape({num_after_axis_dim});
  after_axis_sum_multiplier->set_data_type(in_data_type);
  BlobDesc* before_axis_sum_multiplier =
      GetBlobDesc4BnInOp("before_axis_sum_multiplier");
  before_axis_sum_multiplier->mut_shape() = Shape({num_before_axis_dim});
  before_axis_sum_multiplier->set_data_type(in_data_type);
}

void NormalizationOp::InferParamBlobDescs(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
    const NormalizationOpConf& conf, int64_t norm_part_num,
    DataType in_data_type, bool use_cudnn) const {
  BlobDesc blob_desc(Shape({norm_part_num}), in_data_type, false, false, 1);
  std::list<std::string> blob_names = {"moving_mean", "moving_variance"};
  std::list<std::string> bns_needless_in_predict_or_cudnn = {"new_mean",
                                                             "new_variance"};
  std::list<std::string> bns_need_in_cudnn = {
      "cache_mean_for_cudnn_bw", "cache_inv_variance_for_cudnn_bw"};
  if (conf.center()) {
    blob_names.push_back("beta");
  } else {
    blob_names.push_back("beta_diff");
  }
  if (conf.scale()) {
    blob_names.push_back("gamma");
  } else {
    blob_names.push_back("gamma_diff");
  }
  if (Global<JobDesc>::Get()->IsTrain() && !use_cudnn) {
    for (const std::string& bn : bns_needless_in_predict_or_cudnn) {
      blob_names.push_back(bn);
    }
  }
  if (use_cudnn) {
    for (const std::string& bn : bns_need_in_cudnn) {
      blob_names.push_back(bn);
    }
  } else {
    blob_names.push_back("inv_var");
  }
  for (const auto& bn_in_op : blob_names) {
    *GetBlobDesc4BnInOp(bn_in_op) = blob_desc;
  }
}

void NormalizationOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf,
    const OpContext* op_ctx) const {
  NormalizationKernelConf* conf = kernel_conf->mutable_normalization_conf();
  const auto* ctx = dynamic_cast<const NormalizationOpCtx*>(op_ctx);
  conf->set_use_cudnn(ctx->use_cudnn);
#ifdef WITH_CUDA
  if (ctx->use_cudnn) {
    VirtualGenKernelConfForCudnn(GetBlobDesc4BnInOp, parallel_ctx, kernel_conf);
    return;
  }
#endif
}

#ifdef WITH_CUDA
void NormalizationOp::InferBlobDescsForCudnn(
    std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp) const {
  const auto& conf = op_conf().normalization_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const DataType in_data_type = in_blob_desc->data_type();
  CHECK(conf.scale() && conf.center())
      << "Cudnn batch norm must use scale and center";
  CHECK_GT(conf.epsilon(), CUDNN_BN_MIN_EPSILON);
  InferParamBlobDescs(GetBlobDesc4BnInOp, conf,
                      in_blob_desc->shape().At(conf.axis()), in_data_type,
                      true);
}

void NormalizationOp::VirtualGenKernelConfForCudnn(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  NormalizationKernelConf* conf = kernel_conf->mutable_normalization_conf();
  GetBlobDesc4BnInOp("in")->shape().ToProto(conf->mutable_in());
#if (CUDNN_VERSION >= 7000)
  conf->set_cudnn_bn_mode(CUDNN_BATCHNORM_SPATIAL_PERSISTENT);
#else
  conf->set_cudnn_bn_mode(CUDNN_BATCHNORM_SPATIAL);
#endif
}
#endif

void NormalizationOp::VirtualFixParallelDesc(ParallelDesc* pr_desc) const {
  pr_desc->set_policy(ParallelPolicy::kDataParallel);
}

REGISTER_OP(OperatorConf::kNormalizationConf, NormalizationOp);

}  // namespace oneflow
