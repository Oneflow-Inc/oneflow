#ifndef ONEFLOW_CORE_COMMON_VPU_SCHEDULER_H_
#define ONEFLOW_CORE_COMMON_VPU_SCHEDULER_H_

#include <mutex>
#include "oneflow/core/vm/vpu_instruction.msg.h"

namespace oneflow {

class VpuScheduler final {
 public:
  VpuScheduler(const VpuScheduler&) = default;
  VpuScheduler(VpuScheduler&&) = default;
  VpuScheduler();
  ~VpuScheduler() = default;

  void Receive(OBJECT_MSG_PTR(VpuInstructionMsg) && vpu_instr_msg);

  void MainLoop();

 private:
  void Schedule();

  OBJECT_MSG_PTR(VpuSchedulerCtx) ctx_;
  std::mutex mutex_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_VPU_SCHEDULER_H_
