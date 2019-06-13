#include "oneflow/core/kernel/case_kernel.h"

namespace oneflow {

template<typename T>
void CaseKernel<T>::ForwardDataContent(const KernelCtx& ctx,
                                       std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t cur_selected_id = static_cast<int64_t>(BnInOp2Blob("in")->dptr<T>()[0]);
  *(static_cast<int64_t*>(ctx.other)) = cur_selected_id;
  KernelUtil<DeviceType::kCPU, T>::Set(
      ctx.device_ctx, cur_selected_id,
      BnInOp2Blob(GenRepeatedBn("out", cur_selected_id))->mut_dptr<T>());
}

template<typename T>
void CaseKernel<T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  UNIMPLEMENTED();
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kCaseConf, CaseKernel, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
