#include "oneflow/core/job/machine_context.h"

namespace oneflow {

std::string MachineCtx::GetCtrlAddr(int64_t machine_id) const {
  const Machine& mchn = Global<JobDesc>::Get()->resource().machine(machine_id);
  return mchn.addr() + ":" + std::to_string(mchn.port());
}

MachineCtx::MachineCtx(const std::string& this_mchn_name) {
  this_machine_id_ = Global<JobDesc>::Get()->MachineID4MachineName(this_mchn_name);
  LOG(INFO) << "this machine name: " << this_mchn_name;
  LOG(INFO) << "this machine id: " << this_machine_id_;
#ifdef WITH_CUDA
  ncclGetUniqueId(&nccl_unique_id_);
  ncclGetUniqueId(&nccl_scatter_unique_id_);
  ncclGetUniqueId(&nccl_gather_unique_id_);
#endif
}

}  // namespace oneflow
