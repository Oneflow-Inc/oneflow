#include "oneflow/core/kernel/normalization_kernel.h"
#include "oneflow/core/kernel/normalization_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NormalizationOpConf& conf = this->op_conf().normalization_conf();
  if (conf.scale() && !conf.has_gamma()) {
    InitializerConf gamma_init_conf;
    float gamma_init = conf.gamma_init();
    gamma_init_conf.mutable_constant_conf()->set_value(gamma_init);
    KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &gamma_init_conf, 0,
                                                         BnInOp2Blob("gamma"));
  }
  if (conf.center() && !conf.has_beta()) {
    InitializerConf beta_init_conf;
    float beta_init = conf.beta_init();
    beta_init_conf.mutable_constant_conf()->set_value(beta_init);
    KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &beta_init_conf, 0,
                                                         BnInOp2Blob("beta"));
  }
  if (!conf.has_moving_mean()) {
    float mean_init = conf.mean_init();
    InitializerConf moving_mean_init_conf;
    moving_mean_init_conf.mutable_constant_conf()->set_value(mean_init);
    KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &moving_mean_init_conf, 0,
                                                         BnInOp2Blob("moving_mean"));
  }
  if (!conf.has_moving_variance()) {
    float variance_init = conf.variance_init();
    InitializerConf moving_variance_init_conf;
    moving_variance_init_conf.mutable_constant_conf()->set_value(variance_init);
    KernelUtil<device_type, T>::InitializeWithProperConf(ctx, &moving_variance_init_conf, 0,
                                                         BnInOp2Blob("moving_variance"));
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const NormalizationOpConf& conf = this->op_conf().normalization_conf();
  if (conf.scale() && !conf.has_gamma()) {
    Blob* gamma_blob = BnInOp2Blob("gamma");
    KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir,
                                                  gamma_blob, "gamma", gamma_blob->shape().At(0),
                                                  gamma_blob->shape().Count(1));
  }
  if (conf.center() && !conf.has_beta()) {
    Blob* beta_blob = BnInOp2Blob("beta");
    KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, beta_blob,
                                                  "beta", beta_blob->shape().At(0),
                                                  beta_blob->shape().Count(1));
  }
  if (!conf.has_moving_mean()) {
    Blob* mean_blob = BnInOp2Blob("moving_mean");
    KernelUtil<device_type, T>::InitializeWithDir(ctx, part_id, part_num, model_load_dir, mean_blob,
                                                  "moving_mean", mean_blob->shape().At(0),
                                                  mean_blob->shape().Count(1));
  }
  if (!conf.has_moving_variance()) {
    Blob* variance_blob = BnInOp2Blob("moving_variance");
    KernelUtil<device_type, T>::InitializeWithDir(
        ctx, part_id, part_num, model_load_dir, variance_blob, "moving_variance",
        variance_blob->shape().At(0), variance_blob->shape().Count(1));
  }
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& const_buf_bns = this->op_attribute().const_buf_bns();
  const auto ConstBnExists = [&](const std::string& bn) {
    return std::find(const_buf_bns.cbegin(), const_buf_bns.cend(), bn) != const_buf_bns.cend();
  };
  if (ConstBnExists("beta")) {
    InitializerConf initializer;
    initializer.mutable_constant_conf()->set_value(0);
    KernelUtil<device_type, T>::InitializeWithConf(ctx, initializer, 0, BnInOp2Blob("beta"));
  }
  if (ConstBnExists("gamma")) {
    InitializerConf initializer;
    initializer.mutable_constant_conf()->set_value(1.0);
    KernelUtil<device_type, T>::InitializeWithConf(ctx, initializer, 0, BnInOp2Blob("gamma"));
  }
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
const PbMessage& NormalizationKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().normalization_conf();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kNormalizationConf, NormalizationKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
