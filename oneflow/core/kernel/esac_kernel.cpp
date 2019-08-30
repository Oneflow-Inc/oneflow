#include "oneflow/core/kernel/esac_kernel.h"

namespace oneflow {

template<typename T>
void EsacKernel<T>::ForwardDataContent(const KernelCtx& ctx,
                                       std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  T value = static_cast<T>(*static_cast<int64_t*>(ctx.other));
  KernelUtil<DeviceType::kCPU, T>::Set(ctx.device_ctx, value, BnInOp2Blob("out")->mut_dptr<T>());
}

template<typename T>
void EsacKernel<T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kEsacConf, DeviceType::kCPU, int8_t,
                                      EsacKernel<int8_t>);
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kEsacConf, DeviceType::kCPU, int32_t,
                                      EsacKernel<int32_t>);
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kEsacConf, DeviceType::kCPU, int64_t,
                                      EsacKernel<int64_t>);

}  // namespace oneflow
