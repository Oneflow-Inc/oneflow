#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

std::string RuntimeCtx::GetCtrlAddr(int64_t machine_id) const {
  const Machine& machine = JobDesc::Singleton()->resource().machine(machine_id);
  return machine.addr() + ":" + machine.ctrl_port();
}

void RuntimeCtx::set_this_machine_name(const std::string& name) {
  this_machine_id_ = IDMgr::Singleton()->MachineID4MachineName(name);
  LOG(INFO) << "this machine name: " << name;
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

PersistentInStream* RuntimeCtx::GetDataInStream(const std::string& name) {
  auto it = data_in_streams_.find(name);
  if (it == data_in_streams_.end()) {
    return nullptr;
  } else {
    return it->second.get();
  }
}

void RuntimeCtx::AddDataInStream(const std::string& name,
                                 PersistentInStream* data_in_stream) {
  CHECK(data_in_streams_.find(name) == data_in_streams_.end());
  data_in_streams_[name].reset(data_in_stream);
}

}  // namespace oneflow
