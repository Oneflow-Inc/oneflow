#ifndef ONEFLOW_CORE_COMMON_VPU_SCHEDULER_H_
#define ONEFLOW_CORE_COMMON_VPU_SCHEDULER_H_

#include <mutex>
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/vm/vpu_instruction.msg.h"

namespace oneflow {

using VpuInstructionMsgList = OBJECT_MSG_LIST(VpuInstructionMsg, vpu_instruction_msg_link);

class VpuScheduler final {
 public:
  VpuScheduler(const VpuScheduler&) = default;
  VpuScheduler(VpuScheduler&&) = default;
  explicit VpuScheduler(OBJECT_MSG_PTR(VpuSchedulerCtx) && ctx) : ctx_(std::move(ctx)) {}
  ~VpuScheduler() = default;

  void Receive(VpuInstructionMsgList* vpu_instr_list);

  void Dispatch();

 private:
  OBJECT_MSG_PTR(VpuSchedulerCtx) ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_VPU_SCHEDULER_H_
