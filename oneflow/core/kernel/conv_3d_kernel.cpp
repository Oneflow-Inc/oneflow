#include "oneflow/core/kernel/conv_3d_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConvKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type, typename T>
void ConvKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type, typename T>
void ConvKernel<device_type, T>::InitPureModelTmpBlobs(
    DeviceCtx* ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    InitializerConf bias_multiplier_initializer_conf;
    bias_multiplier_initializer_conf.mutable_constant_conf()->set_value(1.0f);
    KernelUtil<device_type, T>::Initialize(ctx,
                                           bias_multiplier_initializer_conf, 0,
                                           BnInOp2Blob("bias_multiplier"));
  }
}

template<DeviceType device_type, typename T>
void ConvKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    DeviceCtx* ctx, std::mt19937* random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::InitializeWithProperConf(
      ctx,
      static_cast<const InitializerConf*>(
          &this->GetMessageFromCustomizedOpConf("weight_initializer")),
      (*random_seed_gen)(), BnInOp2Blob("weight"));

  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    KernelUtil<device_type, T>::InitializeWithProperConf(
        ctx,
        static_cast<const InitializerConf*>(
            &this->GetMessageFromCustomizedOpConf("bias_initializer")),
        (*random_seed_gen)(), BnInOp2Blob("bias"));
  }
}

template<DeviceType device_type, typename T>
void ConvKernel<device_type, T>::InitModelBlobsWithDir(
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

ADD_DEFAULT_CUDNN_KERNEL_CREATOR(OperatorConf::kConv1DConf, conv_3d_conf,
                                 CudnnConvKernel, FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_CUDNN_KERNEL_CREATOR(OperatorConf::kConv2DConf, conv_3d_conf,
                                 CudnnConvKernel, FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_CUDNN_KERNEL_CREATOR(OperatorConf::kConv3DConf, conv_3d_conf,
                                 CudnnConvKernel, FLOATING_DATA_TYPE_SEQ);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv1DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv2DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv3DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
