#include "oneflow/core/kernel/gather_kernel.h"
#include "oneflow/core/kernel/gather_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& GatherKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().gather_conf();
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("indices");
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  GatherKernelUtil<device_type, T>::Forward(ctx.device_ctx, indices, in,
                                            this->kernel_conf().gather_conf().axis(), out);
}

#define REGISTER_GATHER_KERNELS(dtype)                                                           \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kGatherConf, DeviceType::kCPU, dtype,      \
                                        GatherKernel<DeviceType::kCPU, dtype>)                   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kGatherConf, DeviceType::kGPU, dtype,      \
                                        GatherKernel<DeviceType::kGPU, dtype>)                   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLocalGatherConf, DeviceType::kCPU, dtype, \
                                        GatherKernel<DeviceType::kCPU, dtype>)                   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kLocalGatherConf, DeviceType::kGPU, dtype, \
                                        GatherKernel<DeviceType::kGPU, dtype>)

REGISTER_GATHER_KERNELS(float);
REGISTER_GATHER_KERNELS(double);

}  // namespace oneflow
