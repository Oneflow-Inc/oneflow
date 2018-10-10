#include "oneflow/core/kernel/deconv_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void DeconvKernelIf<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* weight_blob = BnInOp2Blob("weight");
  Blob* out_blob = BnInOp2Blob("out");
  DoForwardDataContent(ctx.device_ctx, in_blob, weight_blob, out_blob, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
void DeconvKernelIf<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* deconv_out_diff = BnInOp2Blob("out_diff");
  if (this->template GetValFromCustomizedOpConf<bool>("use_bias")) {
    BiasBackward(ctx.device_ctx, deconv_out_diff, BnInOp2Blob("bias_diff"), BnInOp2Blob);
  }
  WeightBackward(ctx.device_ctx, deconv_out_diff, BnInOp2Blob("in"), BnInOp2Blob("weight_diff"),
                 BnInOp2Blob("in_diff"), BnInOp2Blob);
}

template<DeviceType device_type, typename T>
const PbMessage& DeconvKernelIf<device_type, T>::GetCustomizedOpConf() const {
  CHECK(this->kernel_conf().has_deconv_conf());
  switch (this->OpKernelDim()) {
    case 2: return this->op_conf().deconv_2d_conf();
    case 3: return this->op_conf().deconv_3d_conf();
    default: UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
const DeconvKernelConf& DeconvKernelIf<device_type, T>::GetDeconvKernelConf() const {
  return this->kernel_conf().deconv_conf();
}

template<DeviceType device_type, typename T>
const int32_t DeconvKernelIf<device_type, T>::OpKernelDim() const {
  return this->GetDeconvKernelConf().dim();
}

template<typename T>
void DeconvKernel<DeviceType::kCPU, T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  UNIMPLEMENTED();
}

template<typename T>
void DeconvKernel<DeviceType::kCPU, T>::DoForwardDataContent(
    DeviceCtx* device_ctx, const Blob* in_blob, const Blob* weight_blob, Blob* out_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<typename T>
void DeconvKernel<DeviceType::kCPU, T>::WeightBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, const Blob* in_blob, Blob* weight_diff_blob,
    Blob* in_diff_blob, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<typename T>
void DeconvKernel<DeviceType::kCPU, T>::BiasBackward(
    DeviceCtx* device_ctx, const Blob* out_diff_blob, Blob* bias_diff_blob,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

#define INSTANTIATE_DECONV_KERNEL_IF(device_type, data_type_pair) \
  template class DeconvKernelIf<device_type, OF_PP_PAIR_FIRST(data_type_pair)>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_DECONV_KERNEL_IF, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ);

#define INSTANTIATE_DECONV_KERNEL(type_cpp, type_proto) \
  template class DeconvKernel<DeviceType::kCPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_DECONV_KERNEL, FLOATING_DATA_TYPE_SEQ)

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kDeconv2DConf, DeconvKernel, FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kDeconv3DConf, DeconvKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
