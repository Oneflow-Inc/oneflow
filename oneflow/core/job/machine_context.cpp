#include "oneflow/core/job/machine_context.h"

namespace oneflow {

std::string MachineCtx::GetCtrlAddr(int64_t machine_id) const {
  const Machine& mchn = JobDesc::Singleton()->resource().machine(machine_id);
  return mchn.addr() + ":" + std::to_string(mchn.port());
}

MachineCtx::MachineCtx(const std::string& this_mchn_name) {
  this_machine_id_ = IDMgr::Singleton()->MachineID4MachineName(this_mchn_name);
  LOG(INFO) << "this machine name: " << this_mchn_name;
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

}  // namespace oneflow
