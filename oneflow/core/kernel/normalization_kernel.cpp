#include "oneflow/core/kernel/normalization_kernel.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/kernel/transpose_kernel.h"
#include "oneflow/core/kernel/normalization_kernel_util.h"

namespace oneflow {

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
      BnInOp2Blob("mean")->dptr<float>(), BnInOp2Blob("inv_variance")->dptr<float>()));
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
void NormalizationKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NormalizationOpConf& conf = this->op_conf().normalization_conf();
  if (this->op_conf().trainable()) {
    NormalizationKernelUtil<device_type, T>::ForwardTraining(
        ctx.device_ctx, BnInOp2Blob("in"), BnInOp2Blob("gamma"), BnInOp2Blob("beta"),
        BnInOp2Blob("out"), BnInOp2Blob("moving_mean"), BnInOp2Blob("moving_variance"),
        BnInOp2Blob("mean"), BnInOp2Blob("inv_variance"), BnInOp2Blob("buf"), conf.axis(),
        conf.epsilon(), conf.momentum());
  } else {
    NormalizationKernelUtil<device_type, T>::ForwardInference(
        ctx.device_ctx, BnInOp2Blob("in"), BnInOp2Blob("gamma"), BnInOp2Blob("beta"),
        BnInOp2Blob("moving_mean"), BnInOp2Blob("moving_variance"), BnInOp2Blob("out"),
        BnInOp2Blob("buf"), conf.axis(), conf.epsilon());
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NormalizationOpConf& conf = this->op_conf().normalization_conf();
  NormalizationKernelUtil<device_type, T>::Backward(
      ctx.device_ctx, BnInOp2Blob("in"), BnInOp2Blob("gamma"), BnInOp2Blob("mean"),
      BnInOp2Blob("inv_variance"), BnInOp2Blob(GenDiffBn("out")), BnInOp2Blob(GenDiffBn("in")),
      BnInOp2Blob(GenDiffBn("gamma")), BnInOp2Blob(GenDiffBn("beta")), BnInOp2Blob("buf"),
      conf.axis(), conf.epsilon());
}

template<DeviceType device_type, typename T>
const PbMessage& NormalizationKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().normalization_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalizationConf, NormalizationKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
