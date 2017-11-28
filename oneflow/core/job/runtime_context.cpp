#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

std::string RuntimeCtx::GetCtrlAddr(int64_t machine_id) const {
  const Machine& mchn = JobDesc::Singleton()->resource().machine(machine_id);
  return mchn.addr() + ":" + std::to_string(mchn.port());
}

RuntimeCtx::RuntimeCtx(const std::string& name) {
  this_machine_id_ = IDMgr::Singleton()->MachineID4MachineName(name);
  LOG(INFO) << "this machine name: " << name;
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

}  // namespace oneflow
