#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

std::string RuntimeCtx::GetCtrlAddr(int64_t machine_id) const {
  const Machine& mchn = JobDesc::Singleton()->resource().machine(machine_id);
  return mchn.addr() + ":" + std::to_string(mchn.port());
}

PersistentOutStream* RuntimeCtx::GetPersistentOutStream(
    const std::string& filepath) {
  auto iter = filepath2ostream_.find(filepath);
  if (iter != filepath2ostream_.end()) {
    return iter->second.get();
  } else {
    auto ostream_ptr = new PersistentOutStream(GlobalFS(), filepath);
    filepath2ostream_[filepath].reset(ostream_ptr);
    return ostream_ptr;
  }
}

RuntimeCtx::RuntimeCtx(const std::string& name) {
  this_machine_id_ = IDMgr::Singleton()->MachineID4MachineName(name);
  LOG(INFO) << "this machine name: " << name;
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

}  // namespace oneflow
