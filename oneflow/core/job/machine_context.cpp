#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

std::string MachineCtx::GetCtrlAddr(int64_t machine_id) const {
  const Machine& mchn = Global<JobDesc>::Get()->resource().machine(machine_id);
  return mchn.addr() + ":" + std::to_string(mchn.port());
}

MachineCtx::MachineCtx() : this_machine_id_(-1) {
  char hostname[32];  // TODO(shiyuan)
  CHECK_EQ(gethostname(hostname, sizeof(hostname)), 0);
  FOR_RANGE(int64_t, i, 0, Global<JobDesc>::Get()->resource().machine_size()) {
    if (std::string(hostname) == Global<JobDesc>::Get()->resource().machine(i).hostname()) {
      this_machine_id_ = Global<JobDesc>::Get()->resource().machine(i).id();
    }
  }
  CHECK_NE(this_machine_id_, -1);
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

}  // namespace oneflow
