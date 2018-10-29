#include "oneflow/core/operator/normalization_op.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

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
  EnrollDataTmpBn("tmp_storage_for_sum");
  EnrollDataTmpBn("trans_in");
  EnrollDataTmpBn("trans_out");
  EnrollDataTmpBn("cache_mean_for_cudnn_bw");
  EnrollDataTmpBn("cache_inv_variance_for_cudnn_bw");
}

const PbMessage& NormalizationOp::GetCustomizedConf() const {
  return op_conf().normalization_conf();
}

void NormalizationOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, std::function<void(OpContext*)> EnrollOpCtx) const {
  const auto& conf = op_conf().normalization_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const DataType in_data_type = in_blob_desc->data_type();
  CHECK_EQ(in_data_type, Global<JobDesc>::Get()->DefaultDataType());
  *GetBlobDesc4BnInOp("out") = *in_blob_desc;
#ifdef WITH_CUDA
  int32_t in_dims = in_blob_desc->shape().NumAxes();
  if (device_type() == DeviceType::kGPU && CUDNN_VERSION >= 5000 && in_data_type == DataType::kFloat
      && in_dims >= 4 && in_dims <= 5 && (conf.axis() == 1 || conf.axis() == in_dims - 1)) {
    InferBlobDescsForCudnn(GetBlobDesc4BnInOp);
    return;
  }
#endif
  NormalizationOpCtx* op_ctx = NewNormalizationOpCtx(in_blob_desc->shape());
  EnrollOpCtx(op_ctx);
  InferParamBlobDescs(GetBlobDesc4BnInOp, conf, op_ctx->transpose_cols, in_data_type, false);
  if (op_ctx->need_transpose) {
    BlobDesc* transpose_blob_desc = GetBlobDesc4BnInOp("trans_in");
    transpose_blob_desc->mut_shape() = in_blob_desc->shape();
    transpose_blob_desc->mut_shape().Set(op_ctx->axis, in_blob_desc->shape().At(0));
    transpose_blob_desc->mut_shape().Set(0, op_ctx->transpose_cols);
    transpose_blob_desc->set_data_type(in_data_type);
    *GetBlobDesc4BnInOp("trans_out") = *transpose_blob_desc;
    *GetBlobDesc4BnInOp("normalized_in") = *transpose_blob_desc;
  } else {
    *GetBlobDesc4BnInOp("normalized_in") = *in_blob_desc;
  }
  size_t tmp_storage_size = 0;
  if (device_type() == DeviceType::kGPU) {
    tmp_storage_size = GetTmpSizeForReduceSum(in_data_type, op_ctx->transpose_rows);
    CHECK_GT(tmp_storage_size, 0);
  }
  BlobDesc* tmp_blob_desc = GetBlobDesc4BnInOp("tmp_storage_for_sum");
  tmp_blob_desc->set_data_type(in_data_type);
  int64_t tmp_elem_cnt =
      static_cast<int64_t>(tmp_storage_size / GetSizeOfDataType(in_data_type)) + 1;
  if (tmp_elem_cnt > 1) { tmp_blob_desc->mut_shape() = Shape({tmp_elem_cnt}); }
}

void NormalizationOp::InferParamBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const NormalizationOpConf& conf, int64_t norm_part_num, DataType in_data_type,
    bool use_cudnn) const {
  BlobDesc blob_desc(Shape({norm_part_num}), in_data_type, false, false, 1);
  std::list<std::string> blob_names = {"moving_mean", "moving_variance"};
  std::list<std::string> bns_needless_in_predict_or_cudnn = {"new_mean", "new_variance"};
  std::list<std::string> bns_need_in_cudnn = {"cache_mean_for_cudnn_bw",
                                              "cache_inv_variance_for_cudnn_bw"};
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
    for (const std::string& bn : bns_needless_in_predict_or_cudnn) { blob_names.push_back(bn); }
  }
  if (use_cudnn) {
    for (const std::string& bn : bns_need_in_cudnn) { blob_names.push_back(bn); }
  } else {
    blob_names.push_back("inv_var");
  }
  for (const auto& bn_in_op : blob_names) { *GetBlobDesc4BnInOp(bn_in_op) = blob_desc; }
}

void NormalizationOp::VirtualGenKernelConf(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf, const OpContext* op_ctx) const {
  NormalizationKernelConf* conf = kernel_conf->mutable_normalization_conf();
  const auto* ctx = dynamic_cast<const NormalizationOpCtx*>(op_ctx);
#ifdef WITH_CUDA
  if (ctx == nullptr) {
    VirtualGenKernelConfForCudnn(GetBlobDesc4BnInOp, parallel_ctx, kernel_conf);
    return;
  }
#endif
  conf->set_axis(ctx->axis);
  conf->set_transpose_rows(ctx->transpose_rows);
  conf->set_transpose_cols(ctx->transpose_cols);
  conf->set_need_transpose(ctx->need_transpose);
  if (ctx->need_transpose) {
    PbRf<int32_t>* perm = conf->mutable_perm();
    perm->Reserve(ctx->dims);
    for (size_t i = 0; i < ctx->dims; ++i) { perm->Add(i); }
    perm->SwapElements(ctx->axis, 0);
  }
  conf->set_use_cudnn(false);
}

#ifdef WITH_CUDA
void NormalizationOp::InferBlobDescsForCudnn(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp) const {
  const auto& conf = op_conf().normalization_conf();
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  const DataType in_data_type = in_blob_desc->data_type();
  CHECK(conf.scale() && conf.center()) << "Cudnn batch norm must use scale and center";
  CHECK_GE(conf.epsilon(), CUDNN_BN_MIN_EPSILON);
  InferParamBlobDescs(GetBlobDesc4BnInOp, conf, in_blob_desc->shape().At(conf.axis()), in_data_type,
                      true);
}

void NormalizationOp::VirtualGenKernelConfForCudnn(
    std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, KernelConf* kernel_conf) const {
  NormalizationKernelConf* conf = kernel_conf->mutable_normalization_conf();
  conf->set_use_cudnn(true);
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

NormalizationOpCtx* NormalizationOp::NewNormalizationOpCtx(const Shape& in_shape) const {
  NormalizationOpCtx* op_ctx = new NormalizationOpCtx();
  op_ctx->axis = op_conf().normalization_conf().axis();
  op_ctx->dims = in_shape.NumAxes();
  if (op_ctx->axis < 0) { op_ctx->axis += op_ctx->dims; }
  CHECK_GE(op_ctx->axis, 0);
  CHECK_LT(op_ctx->axis, op_ctx->dims);
  op_ctx->transpose_cols = in_shape.At(op_ctx->axis);
  op_ctx->transpose_rows = in_shape.elem_cnt() / op_ctx->transpose_cols;
  if (op_ctx->axis == 0) {
    op_ctx->need_transpose = false;
  } else {
    op_ctx->need_transpose = true;
  }
  return op_ctx;
}

REGISTER_OP(OperatorConf::kNormalizationConf, NormalizationOp);

}  // namespace oneflow
