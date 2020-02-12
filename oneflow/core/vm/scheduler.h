#ifndef ONEFLOW_CORE_COMMON_SCHEDULER_H_
#define ONEFLOW_CORE_COMMON_SCHEDULER_H_

#include "oneflow/core/vm/scheduler.msg.h"

namespace oneflow {

using VpuInstructionMsgList = OBJECT_MSG_LIST(VpuInstructionMsg, vpu_instruction_msg_link);

class VpuScheduler final {
 public:
  VpuScheduler(const VpuScheduler&) = default;
  VpuScheduler(VpuScheduler&&) = default;
  explicit VpuScheduler(VpuSchedulerCtx* ctx) : ctx_(ctx) {}
  ~VpuScheduler() = default;

  void Receive(VpuInstructionMsgList* vpu_instr_list);

  void Dispatch();

 private:
  VpuSchedulerCtx* ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SCHEDULER_H_
