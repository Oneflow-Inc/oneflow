#include "oneflow/core/kernel/eltwise_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void EltwiseKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const EltwiseOpConf& eltwise_conf = this->op_conf().eltwise_conf();
  switch (eltwise_conf.operation()) {
    case EltwiseOpConf_EltwiseOp_SUM: break;
    case EltwiseOpConf_EltwiseOp_MAX: break;
    default: break;
  }
}

template<DeviceType device_type>
void EltwiseKernel<device_type>::ForwardDataId(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob(this->kernel_conf().output_bns(0));
  const Blob* in_blob_0 = BnInOp2Blob(this->kernel_conf().input_bns(0));
  out_blob->CopyDataIdFrom(ctx.device_ctx, in_blob_0);
}

template<DeviceType device_type>
void EltwiseKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {}

namespace {

Kernel* CreateEltwiseKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define ELTWISE_KERNEL_ENTRY(device_type) \
  {GetHashKey(device_type), []() { return new EltwiseKernel<device_type>; }},
      OF_PP_FOR_EACH_TUPLE(ELTWISE_KERNEL_ENTRY, DEVICE_TYPE_SEQ)};
  return creators.at(GetHashKey(kernel_conf.device_type()))();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kEltwiseConf, CreateEltwiseKernel));

}  // namespace oneflow
