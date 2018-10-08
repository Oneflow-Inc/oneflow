#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

std::string MachineCtx::GetCtrlAddr(int64_t machine_id) const {
  const Machine& mchn = Global<JobDesc>::Get()->resource().machine(machine_id);
  int32_t host_port = (mchn.rpc_port_agent() != -1) ? (mchn.rpc_port_agent())
                                                    : Global<JobDesc>::Get()->resource().rpc_port();
  return mchn.addr() + ":" + std::to_string(host_port);
}

MachineCtx::MachineCtx(int64_t this_mchn_id) : this_machine_id_(this_mchn_id) {
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

}  // namespace oneflow
