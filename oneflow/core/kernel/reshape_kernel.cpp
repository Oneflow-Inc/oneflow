#include "oneflow/core/kernel/reshape_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReshapeKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");

  Memcpy<device_type>(ctx.device_ctx, out_blob->mut_memory_ptr(),
                      in_blob->memory_ptr(), in_blob->TotalByteSize());
}

template<DeviceType device_type>
void ReshapeKernel<device_type>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_diff_blob = BnInOp2Blob("in_diff");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");

  Memcpy<device_type>(ctx.device_ctx, in_diff_blob->mut_memory_ptr(),
                      out_diff_blob->memory_ptr(),
                      out_diff_blob->TotalByteSize());
}

namespace {

Kernel* CreateReshapeKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
#define RESHAPE_KERNEL_ENTRY(device_type) \
  {GetHashKey(device_type), []() { return new ReshapeKernel<device_type>; }},
      OF_PP_FOR_EACH_TUPLE(RESHAPE_KERNEL_ENTRY, DEVICE_TYPE_SEQ)};
  return creators.at(GetHashKey(kernel_conf.device_type()))();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kReshapeConf, CreateReshapeKernel));

}  // namespace oneflow
