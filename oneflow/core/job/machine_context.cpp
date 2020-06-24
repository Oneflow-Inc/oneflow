#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

std::string MachineCtx::GetCtrlAddr(int64_t machine_id) const {
  const Machine& mchn = Global<ResourceDesc, ForSession>::Get()->machine(machine_id);
  int32_t ctrl_port = (mchn.ctrl_port_agent() != -1) ? (mchn.ctrl_port_agent())
                                                     : Global<EnvDesc>::Get()->ctrl_port();
  return mchn.addr() + ":" + std::to_string(ctrl_port);
}

MachineCtx::MachineCtx(int64_t this_mchn_id) : this_machine_id_(this_mchn_id) {
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

}  // namespace oneflow
