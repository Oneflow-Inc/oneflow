#include "oneflow/core/kernel/normalization_kernel.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/kernel/transpose_kernel.h"

namespace oneflow {

namespace {

InitializerConf ConstantInitializerConf(float val) {
  InitializerConf conf;
  conf.mutable_constant_conf()->set_value(val);
  return conf;
}

InitializerConf OnesInitializerConf() { return ConstantInitializerConf(1.0f); }
InitializerConf ZerosInitializerConf() { return ConstantInitializerConf(0.0f); }

template<DeviceType device_type, typename T>
void Rsqrt(DeviceCtx* ctx, const int64_t n, const T* x, const float epsilon, T* y) {
  KernelUtil<device_type, T>::Copy(ctx, n, x, 1, y, 1);
  KernelUtil<device_type, T>::Rsqrt(ctx, n, y, epsilon);
}

template<DeviceType device_type, typename T>
void ScalarSub(DeviceCtx* ctx, const int64_t n, const T* x, const T* scalar_ptr, T* y) {
  KernelUtil<device_type, T>::Copy(ctx, n, x, 1, y, 1);
  KernelUtil<device_type, T>::Axpy(ctx, n, static_cast<T>(-1), scalar_ptr, 0, y, 1);
}

template<typename T>
T* GetTmpForSumDptr(Blob* tmp_storage_blob, T* nop_addr) {
  if (tmp_storage_blob != nullptr) {
    return tmp_storage_blob->mut_dptr<T>();
  } else {
    return nop_addr;
  }
}
size_t GetTmpForSumByteSize(Blob* tmp_storage_blob) {
  if (tmp_storage_blob != nullptr) {
    return tmp_storage_blob->ByteSizeOfDataContentField();
  } else {
    return 0;
  }
}

}  // namespace

NormalizationCtx::NormalizationCtx(const KernelConf& kernel_conf, DataType type) {
#ifdef WITH_CUDA
  const NormalizationKernelConf& conf = kernel_conf.normalization_conf();
  mode_ = static_cast<cudnnBatchNormMode_t>(conf.cudnn_bn_mode());
  std::vector<int64_t> in_shape(conf.in().dim().begin(), conf.in().dim().end());
  CHECK(4 <= in_shape.size() && in_shape.size() <= 5) << in_shape.size();
  int32_t axis = kernel_conf.op_attribute().op_conf().normalization_conf().axis();
  param_desc_.reset(new CudnnTensorDesc());
  int N = in_shape[0];
  int C = in_shape[axis];
  int H = in_shape[axis == 1 ? 2 : 1];
  int W = in_shape[axis == 1 ? 3 : 2];
  int D = in_shape.size() > 4 ? in_shape[axis == 1 ? 4 : 3] : 1;
  std::vector<int> dims = {N, C, H, W, D};
  std::vector<int> strides;
  if (axis == 1) {
    strides = {C * H * W * D, H * W * D, W * D, D, 1};
  } else {
    strides = {H * W * D * C, 1, W * D * C, D * C, C};
  }
  in_desc_.reset(new CudnnTensorDesc(type, in_shape.size(), dims.data(), strides.data()));
  CudaCheck(cudnnDeriveBNTensorDescriptor(param_desc_->Get(), in_desc_->Get(), mode_));
#endif  // WITH_CUDA
}
#ifdef WITH_CUDA
const cudnnBatchNormMode_t& NormalizationCtx::cudnn_batch_norm_mode() const { return mode_; }
const cudnnTensorDescriptor_t& NormalizationCtx::cudnn_in_tensor_desc() const {
  return in_desc_->Get();
}
const cudnnTensorDescriptor_t& NormalizationCtx::cudnn_param_tensor_desc() const {
  return param_desc_->Get();
}
#endif  // WITH_CUDA

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::NormalizationCudnnForward(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::NormalizationCudnnBackward(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<>
void NormalizationKernel<DeviceType::kGPU, float>::NormalizationCudnnForward(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const float* in = BnInOp2Blob("in")->dptr<float>();
  const float* gamma = BnInOp2Blob("gamma")->dptr<float>();
  const float* beta = BnInOp2Blob("beta")->dptr<float>();
  float* out = BnInOp2Blob("out")->mut_dptr<float>();
  float* moving_mean = BnInOp2Blob("moving_mean")->mut_dptr<float>();
  float* moving_variance = BnInOp2Blob("moving_variance")->mut_dptr<float>();
  double epsilon = this->op_conf().normalization_conf().epsilon();
  if (this->op_conf().trainable()) {
    InitMovingMeanAndMovingVariance(ctx, BnInOp2Blob, false);
    double momentum = this->op_conf().normalization_conf().momentum();
    CudaCheck(cudnnBatchNormalizationForwardTraining(
        ctx.device_ctx->cudnn_handle(), normalization_ctx_->cudnn_batch_norm_mode(),
        OnePtr<float>::value, ZeroPtr<float>::value, normalization_ctx_->cudnn_in_tensor_desc(), in,
        normalization_ctx_->cudnn_in_tensor_desc(), out,
        normalization_ctx_->cudnn_param_tensor_desc(), gamma, beta, 1 - momentum, moving_mean,
        moving_variance, epsilon, BnInOp2Blob("cache_mean_for_cudnn_bw")->mut_dptr<float>(),
        BnInOp2Blob("cache_inv_variance_for_cudnn_bw")->mut_dptr<float>()));
  } else {
    CudaCheck(cudnnBatchNormalizationForwardInference(
        ctx.device_ctx->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, OnePtr<float>::value,
        ZeroPtr<float>::value, normalization_ctx_->cudnn_in_tensor_desc(), in,
        normalization_ctx_->cudnn_in_tensor_desc(), out,
        normalization_ctx_->cudnn_param_tensor_desc(), gamma, beta, moving_mean, moving_variance,
        epsilon));
  }
}

template<>
void NormalizationKernel<DeviceType::kGPU, float>::NormalizationCudnnBackward(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const cudnnTensorDescriptor_t& in_desc = normalization_ctx_->cudnn_in_tensor_desc();
  CudaCheck(cudnnBatchNormalizationBackward(
      ctx.device_ctx->cudnn_handle(), normalization_ctx_->cudnn_batch_norm_mode(),
      OnePtr<float>::value, ZeroPtr<float>::value, OnePtr<float>::value, ZeroPtr<float>::value,
      in_desc, BnInOp2Blob("in")->dptr<float>(), in_desc,
      BnInOp2Blob(GenDiffBn("out"))->dptr<float>(), in_desc,
      BnInOp2Blob(GenDiffBn("in"))->mut_dptr<float>(),
      normalization_ctx_->cudnn_param_tensor_desc(), BnInOp2Blob("gamma")->dptr<float>(),
      BnInOp2Blob(GenDiffBn("gamma"))->mut_dptr<float>(),
      BnInOp2Blob(GenDiffBn("beta"))->mut_dptr<float>(),
      static_cast<double>(this->op_conf().normalization_conf().epsilon()),
      BnInOp2Blob("cache_mean_for_cudnn_bw")->dptr<float>(),
      BnInOp2Blob("cache_inv_variance_for_cudnn_bw")->dptr<float>()));
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& normalization_conf = this->op_conf().normalization_conf();
  if (normalization_conf.scale()) {
    InitializerConf gamma_init_conf;
    float gamma_init = normalization_conf.gamma_init();
    gamma_init_conf.mutable_constant_conf()->set_value(gamma_init);
    KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &gamma_init_conf, 0,
                                                         BnInOp2Blob("gamma"));
  }
  if (normalization_conf.center()) {
    InitializerConf beta_init_conf;
    float beta_init = normalization_conf.beta_init();
    beta_init_conf.mutable_constant_conf()->set_value(beta_init);
    KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &beta_init_conf, 0,
                                                         BnInOp2Blob("beta"));
  }
  float mean_init = normalization_conf.mean_init();
  InitializerConf moving_mean_init_conf;
  moving_mean_init_conf.mutable_constant_conf()->set_value(mean_init);
  KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &moving_mean_init_conf, 0,
                                                       BnInOp2Blob("moving_mean"));
  float variance_init = normalization_conf.variance_init();
  InitializerConf moving_variance_init_conf;
  moving_variance_init_conf.mutable_constant_conf()->set_value(variance_init);
  KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &moving_variance_init_conf, 0,
                                                       BnInOp2Blob("moving_variance"));
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->op_conf().normalization_conf();
  if (conf.scale()) {
    Blob* gamma_blob = BnInOp2Blob("gamma");
    KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir,
                                                  gamma_blob, "gamma", gamma_blob->shape().At(0),
                                                  gamma_blob->shape().Count(1));
  }
  if (conf.center()) {
    Blob* beta_blob = BnInOp2Blob("beta");
    KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, beta_blob,
                                                  "beta", beta_blob->shape().At(0),
                                                  beta_blob->shape().Count(1));
  }
  Blob* mean_blob = BnInOp2Blob("moving_mean");
  KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, mean_blob,
                                                "moving_mean", mean_blob->shape().At(0),
                                                mean_blob->shape().Count(1));
  Blob* variance_blob = BnInOp2Blob("moving_variance");
  KernelUtil<device_type, T>::InitializeWithDir(
      ctx, part_id, part_num, model_load_dir, variance_blob, "moving_variance",
      variance_blob->shape().At(0), variance_blob->shape().Count(1));
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->op_conf().normalization_conf();
  if (!conf.scale()) {
    InitializerConf ones_init = OnesInitializerConf();
    KernelUtil<device_type, T>::InitializeWithConf(ctx, ones_init, 0, BnInOp2Blob("gamma"));
  }
  if (!conf.center()) {
    InitializerConf zero_init = ZerosInitializerConf();
    KernelUtil<device_type, T>::InitializeWithConf(ctx, zero_init, 0, BnInOp2Blob("beta"));
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->kernel_conf().normalization_conf();
#ifdef WITH_CUDA
  if (conf.use_cudnn()) {
    NormalizationCudnnForward(ctx, BnInOp2Blob);
    return;
  }
#endif
  const Blob* mean_blob = nullptr;
  const Blob* variance_blob = nullptr;
  const Blob* comp_in_blob = nullptr;
  Blob* comp_out_blob = nullptr;
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  Blob* trans_in_blob = BnInOp2Blob("trans_in");
  Blob* trans_out_blob = BnInOp2Blob("trans_out");
  if (conf.need_transpose()) {
    Transpose<device_type, T>(ctx.device_ctx, in_blob, trans_in_blob, conf.perm());
    comp_in_blob = trans_in_blob;
    comp_out_blob = trans_out_blob;
  } else {
    comp_in_blob = in_blob;
    comp_out_blob = out_blob;
  }
  if (Global<JobDesc>::Get()->IsTrain()) {
    CalcMeanAndVariance(ctx, BnInOp2Blob, comp_in_blob);
    UpdateMovingMeanAndMovingVariance(ctx, BnInOp2Blob);
    mean_blob = BnInOp2Blob("new_mean");
    variance_blob = BnInOp2Blob("new_variance");
  } else {
    mean_blob = BnInOp2Blob("moving_mean");
    variance_blob = BnInOp2Blob("moving_variance");
  }
  Normalize(ctx, BnInOp2Blob, mean_blob, variance_blob, comp_in_blob, comp_out_blob);
  if (conf.need_transpose()) {
    Transpose<device_type, T>(ctx.device_ctx, trans_out_blob, out_blob, conf.perm());
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& normalization_kernel_conf = this->kernel_conf().normalization_conf();
#ifdef WITH_CUDA
  if (normalization_kernel_conf.use_cudnn()) {
    NormalizationCudnnBackward(ctx, BnInOp2Blob);
    return;
  }
#endif
  const auto& normalization_op_conf = this->op_conf().normalization_conf();
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  const Blob* comp_out_diff_blob = nullptr;
  Blob* comp_in_diff_blob = nullptr;
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  Blob* trans_in_blob = BnInOp2Blob("trans_in");
  Blob* trans_out_blob = BnInOp2Blob("trans_out");
  bool need_transpose = normalization_kernel_conf.need_transpose();
  bool need_comp_in_diff = (in_diff_blob != nullptr);
  if (need_transpose) {
    Transpose<device_type, T>(ctx.device_ctx, out_diff_blob, trans_out_blob,
                              normalization_kernel_conf.perm());
    comp_in_diff_blob = trans_in_blob;
    comp_out_diff_blob = trans_out_blob;
  } else {
    comp_in_diff_blob = in_diff_blob;
    comp_out_diff_blob = out_diff_blob;
  }
  if (need_comp_in_diff || normalization_op_conf.scale()) {
    CalcAboutGammaDiff(ctx, BnInOp2Blob, comp_out_diff_blob, need_comp_in_diff);
  }
  if (need_comp_in_diff || normalization_op_conf.center()) {
    CalcAboutBetaDiff(ctx, BnInOp2Blob, comp_out_diff_blob, need_comp_in_diff);
  }
  if (need_comp_in_diff) {
    CalcInDiff(ctx, BnInOp2Blob, comp_out_diff_blob, comp_in_diff_blob);
    if (need_transpose)
      Transpose<device_type, T>(ctx.device_ctx, trans_in_blob, in_diff_blob,
                                normalization_kernel_conf.perm());
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcAboutGammaDiff(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)> BnInOp2Blob,
    const Blob* out_diff_blob, bool need_comp_in_diff) const {
  const auto& normalization_kernel_conf = this->kernel_conf().normalization_conf();
  const int32_t norm_part_num = normalization_kernel_conf.transpose_cols();
  const int64_t norm_elem_num = normalization_kernel_conf.transpose_rows();
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  Blob* gamma_diff_blob = BnInOp2Blob("gamma_diff");
  Blob* inv_var_blob = BnInOp2Blob("inv_var");
  FOR_RANGE(int32_t, i, 0, norm_part_num) {
    KernelUtil<device_type, T>::Dot(
        ctx.device_ctx, norm_elem_num, out_diff_blob->dptr<T>() + i * norm_elem_num, 1,
        normalized_blob->dptr<T>() + i * norm_elem_num, 1, gamma_diff_blob->mut_dptr<T>() + i);
    if (need_comp_in_diff) {
      KernelUtil<device_type, T>::Scal(ctx.device_ctx, norm_elem_num,
                                       gamma_diff_blob->dptr<T>() + i,
                                       normalized_blob->mut_dptr<T>() + i * norm_elem_num, 1);
    }
  }
  const Blob* gamma_blob = BnInOp2Blob("gamma");
  if (gamma_blob != nullptr) {
    KernelUtil<device_type, T>::Mul(ctx.device_ctx, norm_part_num, gamma_blob->dptr<T>(),
                                    inv_var_blob->dptr<T>(), inv_var_blob->mut_dptr<T>());
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcAboutBetaDiff(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)> BnInOp2Blob,
    const Blob* out_diff_blob, bool need_comp_in_diff) const {
  const auto& normalization_kernel_conf = this->kernel_conf().normalization_conf();
  const int32_t norm_part_num = normalization_kernel_conf.transpose_cols();
  const int64_t norm_elem_num = normalization_kernel_conf.transpose_rows();
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  Blob* beta_diff_blob = BnInOp2Blob("beta_diff");
  Blob* tmp_storage_blob = BnInOp2Blob("tmp_storage_for_sum");
  FOR_RANGE(int32_t, i, 0, norm_part_num) {
    KernelUtil<device_type, T>::Sum(
        ctx.device_ctx, norm_elem_num, out_diff_blob->dptr<T>() + i * norm_elem_num,
        beta_diff_blob->mut_dptr<T>() + i,
        GetTmpForSumDptr<T>(tmp_storage_blob, beta_diff_blob->mut_dptr<T>()),
        GetTmpForSumByteSize(tmp_storage_blob));
    if (need_comp_in_diff) {
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, norm_elem_num, static_cast<T>(1),
                                       beta_diff_blob->dptr<T>() + i, 0,
                                       normalized_blob->mut_dptr<T>() + i * norm_elem_num, 1);
    }
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcInDiff(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)> BnInOp2Blob,
    const Blob* out_diff_blob, Blob* in_diff_blob) const {
  const auto& normalization_kernel_conf = this->kernel_conf().normalization_conf();
  const int32_t norm_part_num = normalization_kernel_conf.transpose_cols();
  const int64_t norm_elem_num = normalization_kernel_conf.transpose_rows();
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  Blob* inv_var_blob = BnInOp2Blob("inv_var");
  KernelUtil<device_type, T>::Scal(ctx.device_ctx, normalized_blob->shape().elem_cnt(),
                                   static_cast<T>(-1.0 / norm_elem_num),
                                   normalized_blob->mut_dptr<T>(), 1);
  KernelUtil<device_type, T>::Axpy(ctx.device_ctx, normalized_blob->shape().elem_cnt(),
                                   static_cast<T>(1), out_diff_blob->dptr<T>(), 1,
                                   normalized_blob->mut_dptr<T>(), 1);
  FOR_RANGE(int32_t, i, 0, norm_part_num) {
    KernelUtil<device_type, T>::Scal(ctx.device_ctx, norm_elem_num, inv_var_blob->dptr<T>() + i,
                                     normalized_blob->mut_dptr<T>() + i * norm_elem_num, 1);
  }
  in_diff_blob->CopyDataContentFrom(ctx.device_ctx, normalized_blob);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::Normalize(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    const Blob* mean_blob, const Blob* variance_blob, const Blob* in_blob, Blob* out_blob) const {
  const auto& normalization_op_conf = this->op_conf().normalization_conf();
  const auto& normalization_kernel_conf = this->kernel_conf().normalization_conf();
  const int32_t norm_part_num = normalization_kernel_conf.transpose_cols();
  const int64_t norm_elem_num = normalization_kernel_conf.transpose_rows();
  const bool scale = normalization_op_conf.scale();
  const bool center = normalization_op_conf.center();
  Blob* inv_var_blob = BnInOp2Blob("inv_var");
  Blob* normalized_blob = BnInOp2Blob("normalized_in");
  Rsqrt<device_type, T>(ctx.device_ctx, norm_part_num, variance_blob->dptr<T>(),
                        normalization_op_conf.epsilon(), inv_var_blob->mut_dptr<T>());
  FOR_RANGE(int32_t, i, 0, norm_part_num) {
    ScalarSub<device_type, T>(ctx.device_ctx, norm_elem_num, in_blob->dptr<T>() + i * norm_elem_num,
                              mean_blob->dptr<T>() + i,
                              normalized_blob->mut_dptr<T>() + i * norm_elem_num);
    KernelUtil<device_type, T>::Scal(ctx.device_ctx, norm_elem_num, inv_var_blob->dptr<T>() + i,
                                     normalized_blob->mut_dptr<T>() + i * norm_elem_num, 1);
  }
  out_blob->CopyDataContentFrom(ctx.device_ctx, normalized_blob);
  if (scale) {
    const Blob* gamma_blob = BnInOp2Blob("gamma");
    FOR_RANGE(int32_t, i, 0, norm_part_num) {
      KernelUtil<device_type, T>::Scal(ctx.device_ctx, norm_elem_num, gamma_blob->dptr<T>() + i,
                                       out_blob->mut_dptr<T>() + i * norm_elem_num, 1);
    }
  }
  if (center) {
    const Blob* beta_blob = BnInOp2Blob("beta");
    FOR_RANGE(int32_t, i, 0, norm_part_num) {
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, norm_elem_num, static_cast<T>(1),
                                       beta_blob->dptr<T>() + i, 0,
                                       out_blob->mut_dptr<T>() + i * norm_elem_num, 1);
    }
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::CalcMeanAndVariance(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    const Blob* in_blob) const {
  const auto& conf = this->kernel_conf().normalization_conf();
  Blob* mean_blob = BnInOp2Blob("new_mean");
  Blob* tmp_storage_blob = BnInOp2Blob("tmp_storage_for_sum");
  const int32_t norm_part_num = conf.transpose_cols();
  const int64_t norm_elem_num = conf.transpose_rows();
  T* tmp_dptr = GetTmpForSumDptr<T>(tmp_storage_blob, mean_blob->mut_dptr<T>());
  size_t tmp_byte_size = GetTmpForSumByteSize(tmp_storage_blob);
  FOR_RANGE(int32_t, i, 0, norm_part_num) {
    KernelUtil<device_type, T>::Sum(ctx.device_ctx, norm_elem_num,
                                    in_blob->dptr<T>() + i * norm_elem_num,
                                    mean_blob->mut_dptr<T>() + i, tmp_dptr, tmp_byte_size);
  }
  const T inv_norm_elem_num = static_cast<T>(1.0 / norm_elem_num);
  KernelUtil<device_type, T>::Scal(ctx.device_ctx, mean_blob->shape().elem_cnt(), inv_norm_elem_num,
                                   mean_blob->mut_dptr<T>(), 1);
  //  It's safe to use `out' as tmp blob
  Blob* tmp_blob = BnInOp2Blob("out");
  Blob* variance_blob = BnInOp2Blob("new_variance");
  FOR_RANGE(int32_t, i, 0, norm_part_num) {
    ScalarSub<device_type, T>(ctx.device_ctx, norm_elem_num, in_blob->dptr<T>() + i * norm_elem_num,
                              mean_blob->dptr<T>() + i, tmp_blob->mut_dptr<T>());
    KernelUtil<device_type, T>::Mul(ctx.device_ctx, norm_elem_num, tmp_blob->dptr<T>(),
                                    tmp_blob->dptr<T>(), tmp_blob->mut_dptr<T>());
    KernelUtil<device_type, T>::Sum(ctx.device_ctx, norm_elem_num, tmp_blob->dptr<T>(),
                                    variance_blob->mut_dptr<T>() + i, tmp_dptr, tmp_byte_size);
  }
  KernelUtil<device_type, T>::Scal(ctx.device_ctx, variance_blob->shape().elem_cnt(),
                                   inv_norm_elem_num, variance_blob->mut_dptr<T>(), 1);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::UpdateMovingMeanAndMovingVariance(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  InitMovingMeanAndMovingVariance(ctx, BnInOp2Blob, true);
  const auto& conf = this->op_conf().normalization_conf();
  const Blob* mean_blob = BnInOp2Blob("new_mean");
  const Blob* variance_blob = BnInOp2Blob("new_variance");
  Blob* moving_mean_blob = BnInOp2Blob("moving_mean");
  Blob* moving_variance_blob = BnInOp2Blob("moving_variance");
  const T momentum = conf.momentum();
  const T one_minus_momentum = 1 - momentum;
  // Do Bessel's correction for new_variance when updated to moving_variance.
  const auto& normalization_kernel_conf = this->kernel_conf().normalization_conf();
  const int64_t norm_elem_num = normalization_kernel_conf.transpose_rows();
  const float correction_factor =
      norm_elem_num > 1 ? 1.0 * norm_elem_num / (norm_elem_num - 1) : 1.0;
  const T variance_momentum = static_cast<T>(one_minus_momentum * correction_factor);

  KernelUtil<device_type, T>::Scal(ctx.device_ctx, moving_mean_blob->shape().elem_cnt(), momentum,
                                   moving_mean_blob->mut_dptr<T>(), 1);
  KernelUtil<device_type, T>::Axpy(ctx.device_ctx, mean_blob->shape().elem_cnt(),
                                   one_minus_momentum, mean_blob->dptr<T>(), 1,
                                   moving_mean_blob->mut_dptr<T>(), 1);
  KernelUtil<device_type, T>::Scal(ctx.device_ctx, moving_variance_blob->shape().elem_cnt(),
                                   momentum, moving_variance_blob->mut_dptr<T>(), 1);
  KernelUtil<device_type, T>::Axpy(ctx.device_ctx, variance_blob->shape().elem_cnt(),
                                   variance_momentum, variance_blob->dptr<T>(), 1,
                                   moving_variance_blob->mut_dptr<T>(), 1);
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitMovingMeanAndMovingVariance(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob,
    bool use_new) const {
  const auto& conf = this->op_conf().normalization_conf();
  auto tpl =
      reinterpret_cast<std::tuple<int64_t, std::function<const Blob*(const LogicalBlobId&)>>*>(
          ctx.other);
  int64_t piece_id = std::get<0>(*tpl);
  std::function<const Blob*(const LogicalBlobId&)> lbi2preblob = std::get<1>(*tpl);
  Blob* moving_mean_blob = BnInOp2Blob("moving_mean");
  Blob* moving_variance_blob = BnInOp2Blob("moving_variance");
  if (this->op_conf().model_load_dir() == ""
      && (Global<SnapshotMgr>::Get() == nullptr
          || Global<SnapshotMgr>::Get()->GetReadableSnapshot() == nullptr)
      && piece_id == 0) {
    if (use_new) {
      if (conf.use_first_piece_init_moving()) {
        const Blob* mean_blob = BnInOp2Blob("new_mean");
        const Blob* variance_blob = BnInOp2Blob("new_variance");
        moving_mean_blob->CopyDataContentFrom(ctx.device_ctx, mean_blob);
        moving_variance_blob->CopyDataContentFrom(ctx.device_ctx, variance_blob);
        return;
      }
    } else {
      Memset<device_type>(ctx.device_ctx, moving_mean_blob->mut_dptr<T>(), 0,
                          moving_mean_blob->ByteSizeOfDataContentField());
      Memset<device_type>(ctx.device_ctx, moving_variance_blob->mut_dptr<T>(), 0,
                          moving_mean_blob->ByteSizeOfDataContentField());
      return;
    }
  }
  const Blob* pre_moving_mean_blob = lbi2preblob(this->BnInOp2Lbi("moving_mean"));
  if (pre_moving_mean_blob != moving_mean_blob) {
    moving_mean_blob->CopyDataContentFrom(ctx.device_ctx, pre_moving_mean_blob);
  }
  const Blob* pre_moving_variance_blob = lbi2preblob(this->BnInOp2Lbi("moving_variance"));
  if (pre_moving_variance_blob != moving_variance_blob) {
    moving_variance_blob->CopyDataContentFrom(ctx.device_ctx, pre_moving_variance_blob);
  }
}

template<DeviceType device_type, typename T>
const PbMessage& NormalizationKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().normalization_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalizationConf, NormalizationKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
