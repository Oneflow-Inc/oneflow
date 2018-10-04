#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

std::string MachineCtx::GetCtrlAddr(int64_t machine_id) const {
  const Machine& mchn = Global<JobDesc>::Get()->resource().machine(machine_id);
  return mchn.port_map() == 0 ? (mchn.addr() + ":" + std::to_string(mchn.port()))
                              : (mchn.addr() + ":" + std::to_string(mchn.port_map()));
}

std::string MachineCtx::GetCtrlServerAddr(int64_t machine_id) const {
  const Machine& mchn = Global<JobDesc>::Get()->resource().machine(machine_id);
  return "0.0.0.0:" + std::to_string(mchn.port());
  // return mchn.addr() + ":" + std::to_string(mchn.port());
}

MachineCtx::MachineCtx(int64_t this_mchn_id) : this_machine_id_(this_mchn_id) {
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

}  // namespace oneflow
