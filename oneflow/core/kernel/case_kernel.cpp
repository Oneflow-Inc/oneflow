#include "oneflow/core/kernel/case_kernel.h"

namespace oneflow {

template<typename T>
void CaseKernel<T>::ForwardDataContent(const KernelCtx& ctx,
                                       std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CaseStatus* const case_status = static_cast<CaseStatus*>(ctx.other);
  if (case_status->cmd == kCaseCmdHandleInput) {
    int64_t cur_selected_id = static_cast<int64_t>(BnInOp2Blob("in")->dptr<T>()[0]);
    case_status->select_id2request_cnt[cur_selected_id] += 1;
  } else if (case_status->cmd == kCaseCmdHandleOutput) {
    int64_t cur_selected_id = case_status->cur_selected_id;
    CHECK_GT(case_status->select_id2request_cnt[cur_selected_id], 0);
    case_status->select_id2request_cnt[cur_selected_id] -= 1;
    if (case_status->select_id2request_cnt[cur_selected_id] == 0) {
      case_status->select_id2request_cnt.erase(cur_selected_id);
    }
    KernelUtil<DeviceType::kCPU, T>::Set(
        ctx.device_ctx, cur_selected_id,
        BnInOp2Blob(GenRepeatedBn("out", cur_selected_id))->mut_dptr<T>());
  } else {
    UNIMPLEMENTED();
  }
}

REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kCaseConf, DeviceType::kCPU, int8_t,
                                      CaseKernel<int8_t>);
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kCaseConf, DeviceType::kCPU, int32_t,
                                      CaseKernel<int32_t>);
REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kCaseConf, DeviceType::kCPU, int64_t,
                                      CaseKernel<int64_t>);

}  // namespace oneflow
