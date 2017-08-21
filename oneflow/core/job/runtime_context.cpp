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

void RuntimeCtx::AddLocalNetMemoryDesc(
    const NetMemoryDescriptor& net_memory_desc) {
  std::lock_guard<std::mutex> lock(mutex_);
  local_net_memory_descs_.push_back(net_memory_desc);
}

const std::vector<NetMemoryDescriptor>& RuntimeCtx::local_net_memory_descs()
    const {
  return local_net_memory_descs_;
}

void RuntimeCtx::AddRegst2NetMemory(void* regst, void* net_memory) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK(regst2net_memory_.insert({regst, net_memory}).second);
}

void* RuntimeCtx::net_memory_from_regst(void* regst) {
  auto net_memory_it = regst2net_memory_.find(regst);
  CHECK(net_memory_it != regst2net_memory_.end());
  return net_memory_it->second;
}

void RuntimeCtx::AddRemoteMemoryDescriptor(
    int64_t machine_id, const RemoteRegstDesc& remote_regst_desc) {
  MemoryDescriptor mem_desc;
  mem_desc.address = remote_regst_desc.data_address();
  mem_desc.machine_id = machine_id;
  mem_desc.remote_token = remote_regst_desc.remote_token();

  int64_t consumer_task_id = remote_regst_desc.consumer_task_id();
  uint64_t regst_address = remote_regst_desc.regst_address();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    CHECK(remote_net_memory_descs_
              .insert({{consumer_task_id, regst_address}, mem_desc})
              .second);
  }
  return;
}

const MemoryDescriptor& RuntimeCtx::memory_descriptor(
    int64_t consumer_task_id, uint64_t regst_address) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto key = std::make_pair(consumer_task_id, regst_address);
  auto mem_desc_it = remote_net_memory_descs_.find(key);
  CHECK(mem_desc_it != remote_net_memory_descs_.end());
  return mem_desc_it->second;
}

}  // namespace oneflow
