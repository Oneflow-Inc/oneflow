#include "oneflow/core/kernel/gather_ms0_grad_kernel.h"
#include "oneflow/core/kernel/gather_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& GatherMs0GradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().gather_ms0_grad_conf();
}

template<DeviceType device_type, typename T>
void GatherMs0GradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("indices");
  const Blob* out_diff = BnInOp2Blob("out_diff");
  Blob* in_diff = BnInOp2Blob("in_diff");
  const int64_t offset = this->kernel_conf().gather_ms0_grad_conf().offset();
  Memset<device_type>(ctx.device_ctx, in_diff->mut_dptr<T>(), 0,
                      in_diff->ByteSizeOfDataContentField());
  GatherKernelUtil<device_type, T>::Backward(ctx.device_ctx, indices, out_diff, 0, in_diff, offset);
}

REGISTER_KERNEL_HELPER_GPU_FLOATING(OperatorConf::kGatherMs0GradConf, GatherMs0GradKernel);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
REGISTER_KERNEL_HELPER_GPU_HALF(OperatorConf::kGatherMs0GradConf, GatherMs0GradKernel);
#endif

}  // namespace oneflow
