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

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("indices");
  const Blob* out_diff = BnInOp2Blob(GenDiffBn("out"));
  Blob* in_diff = BnInOp2Blob(GenDiffBn("in"));
  Memset<device_type>(ctx.device_ctx, in_diff->mut_dptr<T>(), 0,
                      in_diff->ByteSizeOfDataContentField());
  GatherKernelUtil<device_type, T>::Backward(ctx.device_ctx, indices, out_diff,
                                             this->kernel_conf().gather_conf().axis(), in_diff);
}

namespace {

Kernel* CreateGatherKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
    OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY, (GatherKernel), DEVICE_TYPE_SEQ,
                                     FLOATING_DATA_TYPE_SEQ)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        MAKE_KERNEL_CREATOR_ENTRY(GatherKernel, DeviceType::kGPU, (float16, DataType::kFloat16))
#endif
  };
  return creators.at(
      GetHashKey(kernel_conf.op_attribute().op_conf().device_type(), kernel_conf.data_type()))();
}

REGISTER_KERNEL_CREATOR(OperatorConf::kGatherConf, CreateGatherKernel);
}  // namespace

}  // namespace oneflow
