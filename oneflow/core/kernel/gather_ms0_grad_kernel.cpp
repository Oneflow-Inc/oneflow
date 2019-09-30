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
  Memset<device_type>(ctx.device_ctx, in_diff->mut_dptr<T>(), 0, in_diff->ByteSizeOfBlobBody());
  GatherKernelUtil<device_type, T>::Backward(ctx.device_ctx, indices, out_diff, 0, in_diff, offset);
}

namespace {

Kernel* CreateGatherGradKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
    OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY, (GatherMs0GradKernel),
                                     DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        MAKE_KERNEL_CREATOR_ENTRY(GatherMs0GradKernel, DeviceType::kGPU,
                                  (float16, DataType::kFloat16))
#endif
  };
  return creators.at(
      GetHashKey(kernel_conf.op_attribute().op_conf().device_type(), kernel_conf.data_type()))();
}

REGISTER_KERNEL_CREATOR(OperatorConf::kGatherMs0GradConf, CreateGatherGradKernel);
}  // namespace

}  // namespace oneflow
