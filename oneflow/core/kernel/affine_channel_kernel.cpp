#include "oneflow/core/kernel/affine_channel_kernel.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/kernel/transpose_kernel.h"

namespace oneflow {

AffineChannelCtx::AffineChannelCtx(const KernelConf& kernel_conf, DataType type) {
#ifdef WITH_CUDA
  const AffineChannelKernelConf& conf = kernel_conf.affine_channel_conf();
  mode_ = static_cast<cudnnBatchNormMode_t>(conf.cudnn_bn_mode());
  std::vector<int64_t> in_shape(conf.in().dim().begin(), conf.in().dim().end());
  CHECK(4 <= in_shape.size() && in_shape.size() <= 5) << in_shape.size();
  int32_t axis = kernel_conf.op_attribute().op_conf().affine_channel_conf().axis();
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
const cudnnBatchNormMode_t& AffineChannelCtx::cudnn_batch_norm_mode() const { return mode_; }
const cudnnTensorDescriptor_t& AffineChannelCtx::cudnn_in_tensor_desc() const {
  return in_desc_->Get();
}
const cudnnTensorDescriptor_t& AffineChannelCtx::cudnn_param_tensor_desc() const {
  return param_desc_->Get();
}
#endif  // WITH_CUDA

template<DeviceType device_type, typename T>
void AffineChannelKernel<device_type, T>::NormalizationCudnnForward(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type, typename T>
void AffineChannelKernel<device_type, T>::NormalizationCudnnBackward(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<>
void AffineChannelKernel<DeviceType::kGPU, float>::NormalizationCudnnForward(
    const KernelCtx& ctx, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const float* in = BnInOp2Blob("in")->dptr<float>();
  const float* gamma = BnInOp2Blob("gamma")->dptr<float>();
  const float* beta = BnInOp2Blob("beta")->dptr<float>();
  float* out = BnInOp2Blob("out")->mut_dptr<float>();
  const float* moving_mean = BnInOp2Blob("moving_mean")->dptr<float>();
  const float* moving_variance = BnInOp2Blob("moving_variance")->dptr<float>();
  CudaCheck(cudnnBatchNormalizationForwardInference(
      ctx.device_ctx->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, OnePtr<float>::value,
      ZeroPtr<float>::value, affine_channel_ctx_->cudnn_in_tensor_desc(), in,
      affine_channel_ctx_->cudnn_in_tensor_desc(), out,
      affine_channel_ctx_->cudnn_param_tensor_desc(), gamma, beta, moving_mean, moving_variance,
      OneVal<double>::value));
}

template<DeviceType device_type, typename T>
void AffineChannelKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& affine_channel_conf = this->op_conf().affine_channel_conf();
  if (affine_channel_conf.scale()) {
    InitializerConf gamma_init_conf;
    float gamma_init = affine_channel_conf.gamma_init();
    gamma_init_conf.mutable_constant_conf()->set_value(gamma_init);
    KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &gamma_init_conf, 0,
                                                         BnInOp2Blob("gamma"));
  }
  if (affine_channel_conf.center()) {
    InitializerConf beta_init_conf;
    float beta_init = affine_channel_conf.beta_init();
    beta_init_conf.mutable_constant_conf()->set_value(beta_init);
    KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &beta_init_conf, 0,
                                                         BnInOp2Blob("beta"));
  }
}

template<DeviceType device_type, typename T>
void AffineChannelKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->op_conf().affine_channel_conf();
  if (conf.scale()) {
    Blob* gamma_blob = BnInOp2Blob("gamma");
    KernelUtil<device_type, T>::InitializeWithDir(ctx, 0, part_num, model_load_dir, gamma_blob,
                                                  "gamma", gamma_blob->shape().At(0),
                                                  gamma_blob->shape().Count(1));
  }
  if (conf.center()) {
    Blob* beta_blob = BnInOp2Blob("beta");
    KernelUtil<device_type, T>::InitializeWithDir(ctx, 0, part_num, model_load_dir, beta_blob,
                                                  "beta", beta_blob->shape().At(0),
                                                  beta_blob->shape().Count(1));
  }
}

template<DeviceType device_type, typename T>
void AffineChannelKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* mean_blob = BnInOp2Blob("moving_mean");
  Blob* variance_blob = BnInOp2Blob("moving_variance");
  Memset<device_type>(ctx, mean_blob->mut_dptr<T>(), 0, mean_blob->ByteSizeOfDataContentField());
  Memset<device_type>(ctx, variance_blob->mut_dptr<T>(), 0,
                      variance_blob->ByteSizeOfDataContentField());
}

template<DeviceType device_type, typename T>
void AffineChannelKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto& conf = this->kernel_conf().affine_channel_conf();
#ifdef WITH_CUDA
  if (conf.use_cudnn()) {
    NormalizationCudnnForward(ctx, BnInOp2Blob);
    return;
  }
#endif
  UNIMPLEMENTED();
}

template<DeviceType device_type, typename T>
void AffineChannelKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type, typename T>
const PbMessage& AffineChannelKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().affine_channel_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kAffineChannelConf, AffineChannelKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
