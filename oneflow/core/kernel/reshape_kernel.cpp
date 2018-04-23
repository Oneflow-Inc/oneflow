#include "oneflow/core/kernel/reshape_kernel.h"

namespace oneflow {

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
