#ifndef ONEFLOW_CORE_KERNEL_CASE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CASE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

enum CaseCmd {
  kCaseCmdInvalid = 0,
  kCaseCmdHandleInput = 1,
  kCaseCmdHandleOutput = 2,
};

struct CaseStatus final {
  CaseStatus(): cmd(kCaseCmdInvalid), cur_selected_id(-1) {}
  ~CaseStatus() = default;

  CaseCmd cmd;
  int64_t cur_selected_id;
  HashMap<int64_t, int64_t> select_id2request_cnt;
};

template<typename T>
class CaseKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CaseKernel);
  CaseKernel() = default;
  ~CaseKernel() override = default;

  void Init(const KernelConf& kernel_conf);

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CASE_KERNEL_H_
