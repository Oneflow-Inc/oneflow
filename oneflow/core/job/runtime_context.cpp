#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

std::string RuntimeCtx::GetAddr(int64_t machine_id) const {
  return JobDesc::Singleton()->resource().machine(machine_id).addr();
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

RuntimeCtx::RuntimeCtx(const std::string& name) {
  this_machine_id_ = IDMgr::Singleton()->MachineID4MachineName(name);
  LOG(INFO) << "this machine name: " << name;
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

}  // namespace oneflow
