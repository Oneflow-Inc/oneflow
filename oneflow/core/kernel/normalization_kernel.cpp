#include "oneflow/core/kernel/normalization_kernel.h"
#include "oneflow/core/kernel/normalization_kernel_util.h"

namespace oneflow {

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
