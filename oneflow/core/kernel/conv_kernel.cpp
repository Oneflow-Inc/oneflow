#include "oneflow/core/kernel/conv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  WeightForward(ctx.device_ctx, BnInOp2Blob("in"), BnInOp2Blob("weight"),
                BnInOp2Blob("out"), BnInOp2Blob);
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    BiasForward(ctx.device_ctx, BnInOp2Blob("bias"), BnInOp2Blob("out"));
  }
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    BiasBackward(ctx.device_ctx, BnInOp2Blob("out_diff"),
                 BnInOp2Blob("bias_diff"));
  }
  WeightBackward(ctx.device_ctx, BnInOp2Blob("out_diff"), BnInOp2Blob("in"),
                 BnInOp2Blob("weight_diff"), BnInOp2Blob);
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob) {
    DataBackward(ctx.device_ctx, BnInOp2Blob("out_diff"), BnInOp2Blob("weight"),
                 in_diff_blob, BnInOp2Blob);
  }
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::InitPureModelTmpBlobs(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->GetBoolFromCustomizedOpConf("use_bias") && !this->UseCudnn()) {
    InitializerConf bias_multiplier_initializer_conf;
    bias_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
    KernelUtil<device_type, T>::Initialize(ctx,
                                           bias_multiplier_initializer_conf, 0,
                                           BnInOp2Blob("bias_multiplier"));
  }
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      GetMsgPtrFromPbMessage(this->GetCustomizedOpConf(), "weight_initializer"),
      (*random_seed_gen)(), BnInOp2Blob("weight"));

  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx,
        GetMsgPtrFromPbMessage(this->GetCustomizedOpConf(), "bias_initializer"),
        (*random_seed_gen)(), BnInOp2Blob("bias"));
  }
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::InitModelBlobsWithDir(
    DeviceCtx* ctx, int32_t part_id, int32_t part_num,
    const std::string& model_load_dir,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_blob = BnInOp2Blob("weight");
  int32_t dim_num = this->GetInt32FromCustomizedOpConf("filters");
  KernelUtil<device_type, T>::InitializeWithModelDir(
      ctx, part_id, part_num, model_load_dir, weight_blob, "weight", dim_num,
      weight_blob->shape().Count(1));
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    KernelUtil<device_type, T>::InitializeWithModelDir(
        ctx, part_id, part_num, model_load_dir, BnInOp2Blob("bias"), "bias",
        dim_num, 1);
  }
}

template<DeviceType device_type, typename T>
const PbMessage& ConvKernelIf<device_type, T>::GetCustomizedOpConf() const {
  CHECK(this->kernel_conf().has_conv_conf());
  switch (OpKernelDim()) {
    case 1: return this->op_conf().conv_1d_conf();
    case 2: return this->op_conf().conv_2d_conf();
    case 3: return this->op_conf().conv_3d_conf();
    default: UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
const ConvKernelConf& ConvKernelIf<device_type, T>::GetConvKernelConf() const {
  return this->kernel_conf().conv_conf();
}

template<DeviceType device_type, typename T>
const int32_t ConvKernelIf<device_type, T>::OpKernelDim() const {
  return this->GetConvKernelConf().in().dim_size() - 2;
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::WeightForward(
    DeviceCtx* device_ctx, const Blob* in, const Blob* weight, Blob* out,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::BiasForward(DeviceCtx* device_ctx,
                                                  const Blob* bias,
                                                  Blob* out) const {
  UNIMPLEMENTED();
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::DataBackward(
    DeviceCtx* device_ctx, const Blob* out_diff, const Blob* weight,
    Blob* in_diff, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::WeightBackward(
    DeviceCtx* device_ctx, const Blob* out_diff, const Blob* in,
    Blob* weight_diff,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::BiasBackward(DeviceCtx* device_ctx,
                                                   const Blob* out_diff,
                                                   Blob* bias_diff) const {
  UNIMPLEMENTED();
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv1DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv2DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv3DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
