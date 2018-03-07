#include "oneflow/core/kernel/conv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  VirtualWeightForward(ctx.device_ctx, BnInOp2Blob("in"), BnInOp2Blob("weight"),
                       BnInOp2Blob("out"), BnInOp2Blob("cudnn_workspace"));
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    VirtualBiasForward(ctx.device_ctx, BnInOp2Blob("bias"), BnInOp2Blob("out"));
  }
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (this->GetBoolFromCustomizedOpConf("use_bias")) {
    Blob* bias_diff_blob = BnInOp2Blob("bias_diff");
    Memset<device_type>(ctx.device_ctx, bias_diff_blob->mut_dptr(), 0,
                        bias_diff_blob->ByteSizeOfDataContentField());
    VirtualBiasBackward(ctx.device_ctx, BnInOp2Blob("out_diff"),
                        bias_diff_blob);
  }

  Blob* weight_diff_blob = BnInOp2Blob("weight_diff");
  Memset<device_type>(ctx.device_ctx, weight_diff_blob->mut_dptr(), 0,
                      weight_diff_blob->ByteSizeOfDataContentField());
  VirtualWeightBackward(ctx.device_ctx, BnInOp2Blob("out_diff"),
                        BnInOp2Blob("in"), weight_diff_blob,
                        BnInOp2Blob("cudnn_workspace"));

  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  if (in_diff_blob == nullptr) { return; }
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  VirtualDataBackward(ctx.device_ctx, BnInOp2Blob("out_diff"),
                      BnInOp2Blob("weight"), in_diff_blob,
                      BnInOp2Blob("cudnn_workspace"));
}

template<DeviceType device_type, typename T>
void ConvKernelIf<device_type, T>::InitPureModelTmpBlobs(
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
void ConvKernelIf<device_type, T>::InitModelBlobsWithRandomSeed(
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
  switch (KernelDim()) {
    case 1: return this->op_conf().conv_1d_conf();
    case 2: return this->op_conf().conv_2d_conf();
    case 3: return this->op_conf().conv_3d_conf();
    default: UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
const PbMessage& ConvKernelIf<device_type, T>::GetCustomizedKernelConf() const {
  return this->kernel_conf().conv_conf();
}

template<DeviceType device_type, typename T>
const int32_t ConvKernelIf<device_type, T>::KernelDim() const {
  Shape in_blob_shape(static_cast<const ShapeProto&>(
      this->GetMessageFromCustomizedKernelConf("in")));
  return in_blob_shape.NumAxes() - 2;
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::VirtualWeightForward(
    DeviceCtx* device_ctx, const Blob* in_blob, const Blob* weight_blob,
    Blob* out_blob, Blob* cudnn_workspace) const {
  UNIMPLEMENTED();
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::VirtualBiasForward(DeviceCtx* device_ctx,
                                                         const Blob* bias_blob,
                                                         Blob* out_blob) const {
  UNIMPLEMENTED();
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::VirtualDataBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, const Blob* weight_blob,
    Blob* in_diff_blob, Blob* cudnn_workspace) const {
  UNIMPLEMENTED();
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::VirtualWeightBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, const Blob* in_blob,
    Blob* weight_diff_blob, Blob* cudnn_workspace) const {
  UNIMPLEMENTED();
}

template<typename T>
void ConvKernel<DeviceType::kCPU, T>::VirtualBiasBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob,
    Blob* bias_diff_blob) const {
  UNIMPLEMENTED();
}

#define INSTANTIATE_CONV_KERNEL(type_cpp, type_proto) \
  template class ConvKernel<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_CONV_KERNEL, FLOATING_DATA_TYPE_SEQ);

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv1DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv2DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kConv3DConf, ConvKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
