#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void RuntimeCtx::set_this_machine_name(const std::string& name) {
  this_machine_name_ = name;
  this_machine_id_ = IDMgr::Singleton()->MachineID4MachineName(name);
  LOG(INFO) << "this machine name: " << this_machine_name_;
  LOG(INFO) << "this machine id: " << this_machine_id_;
}

void RuntimeCtx::InitDataReader(const std::string& filepath) {
  data_reader_.reset(new PersistentCircularLineReader(filepath));
}

void RuntimeCtx::AddNetMemoryDesc(const NetMemoryDescriptor& net_memory_desc) {
  std::lock_guard<std::mutex> lock(mutex_);
  net_memory_descs_.push_back(net_memory_desc);
}

const std::vector<NetMemoryDescriptor>& RuntimeCtx::net_memory_descs() const {
  return net_memory_descs_;
}

}  // namespace oneflow
