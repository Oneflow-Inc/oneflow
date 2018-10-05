#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

std::string MachineCtx::GetCtrlAddr(int64_t machine_id) const {
  const Machine& mchn = Global<JobDesc>::Get()->resource().machine(machine_id);
  return mchn.addr();
}

std::string MachineCtx::GetCtrlPort(int64_t machine_id) const {
  const Machine& mchn = Global<JobDesc>::Get()->resource().machine(machine_id);
  return mchn.external_port() == 0 ? std::to_string(mchn.port())
                                   : std::to_string(mchn.external_port());
}

std::string MachineCtx::GetListenPort(int64_t machine_id) const {
  const Machine& mchn = Global<JobDesc>::Get()->resource().machine(machine_id);
  return std::to_string(mchn.port());
}

MachineCtx::MachineCtx(int64_t this_mchn_id) : this_machine_id_(this_mchn_id) {
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

}  // namespace oneflow
