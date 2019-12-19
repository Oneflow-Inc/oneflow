#include "oneflow/core/kernel/esac_kernel.h"

namespace oneflow {

template<typename T>
void EsacKernel<T>::ForwardDataContent(const KernelCtx& ctx,
                                       std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  T value = static_cast<T>(*static_cast<int64_t*>(ctx.other));
  KernelUtil<DeviceType::kCPU, T>::Set(ctx.device_ctx, value, BnInOp2Blob("out")->mut_dptr<T>());
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kEsacConf, EsacKernel, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
