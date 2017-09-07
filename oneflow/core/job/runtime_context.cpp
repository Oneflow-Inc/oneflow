#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void RuntimeCtx::set_this_machine_name(const std::string& name) {
  this_machine_name_ = name;
  this_machine_id_ = IDMgr::Singleton()->MachineID4MachineName(name);
  LOG(INFO) << "this machine name: " << this_machine_name_;
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

DataReader* RuntimeCtx::GetDataReader(const std::string& name) {
  auto it = data_reader_.find(name);
  if (it == data_reader_.end()) {
    return nullptr;
  } else {
    return it->second.get();
  }
}

void RuntimeCtx::AddDataReader(const std::string& name,
                               DataReader* data_reader) {
  CHECK(data_reader_.find(name) == data_reader_.end());
  data_reader_[name].reset(data_reader);
}

}  // namespace oneflow
