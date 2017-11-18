#include "oneflow/core/job/id_manager.h"

namespace oneflow {

int64_t IDMgr::MachineID4MachineName(const std::string& machine_name) const {
  return machine_name2machine_id_.at(machine_name);
}
const std::string& IDMgr::MachineName4MachineId(int64_t machine_id) const {
  return machine_id2machine_name_.at(machine_id);
}

DeviceType IDMgr::GetDeviceTypeFromThrdLocId(int64_t thrd_loc_id) const {
  if (thrd_loc_id < device_num_per_machine_) {
    return JobDesc::Singleton()->resource().device_type();
  } else {
    return DeviceType::kCPU;
  }
}

int64_t IDMgr::NewTaskId(int64_t machine_id, int64_t thrd_local_id) {
  int64_t machine_id64bit = machine_id << (63 - machine_id_bit_num_);
  int64_t device_id64bit = thrd_local_id << task_id_bit_num_;
  int64_t thrd_id = machine_id64bit | device_id64bit;
  CHECK_LT(thread_id2num_of_tasks_[thrd_id],
           (static_cast<int64_t>(1) << task_id_bit_num_) - 1);
  return thrd_id | (thread_id2num_of_tasks_[thrd_id]++);
}

int64_t IDMgr::MachineId4ActorId(int64_t actor_id) const {
  return actor_id >> (63 - machine_id_bit_num_);
}

int64_t IDMgr::ThrdLocId4ActorId(int64_t actor_id) const {
  int64_t tmp = (actor_id << machine_id_bit_num_);
  tmp &= ~(static_cast<int64_t>(1) << 63);
  return tmp >> (machine_id_bit_num_ + task_id_bit_num_);
}

IDMgr::IDMgr() {
  const Resource& resource = JobDesc::Singleton()->resource();
  machine_num_ = resource.machine_size();
  CHECK_LT(machine_num_, static_cast<int64_t>(1) << machine_id_bit_num_);
  device_num_per_machine_ = resource.device_num_per_machine();
  CHECK_LT(device_num_per_machine_,
           (static_cast<int64_t>(1) << device_id_bit_num_) - 3);
  for (int64_t i = 0; i < machine_num_; ++i) {
    const std::string& machine_name = resource.machine(i).name();
    CHECK(machine_name2machine_id_.emplace(machine_name, i).second);
    CHECK(machine_id2machine_name_.emplace(i, machine_name).second);
  }
  regst_desc_id_count_ = 0;
}

}  // namespace oneflow
