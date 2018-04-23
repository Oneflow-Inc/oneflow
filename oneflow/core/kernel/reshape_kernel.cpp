#include "oneflow/core/kernel/reshape_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReshapeKernel<device_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->CopyFrom(ctx.device_ctx, BnInOp2Blob("in"));
}

template<DeviceType device_type>
void ReshapeKernel<device_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("in_diff")->CopyFrom(ctx.device_ctx, BnInOp2Blob("out_diff"));
}

namespace {

Kernel* CreateReshapeKernel(const KernelConf& kernel_conf) {
  static const HashMap<int32_t, std::function<Kernel*()>> creators = {
#define RESHAPE_KERNEL_ENTRY(device_type) \
  {device_type, []() { return new ReshapeKernel<device_type>; }},
      OF_PP_FOR_EACH_TUPLE(RESHAPE_KERNEL_ENTRY, DEVICE_TYPE_SEQ)};
  return creators.at(kernel_conf.device_type())();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kReshapeConf, CreateReshapeKernel));

}  // namespace oneflow
