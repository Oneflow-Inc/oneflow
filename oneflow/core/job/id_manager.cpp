#include "oneflow/core/job/id_manager.h"

namespace oneflow {

int64_t IDMgr::MachineID4MachineName(const std::string& machine_name) const {
  return machine_name2machine_id_.at(machine_name);
}
const std::string& IDMgr::MachineName4MachineId(int64_t machine_id) const {
  return machine_id2machine_name_.at(machine_id);
}

DeviceType IDMgr::GetDeviceTypeFromThrdId(int64_t thrd_id) const {
  if (thrd_id < device_num_per_machine_) {
    return JobDesc::Singleton()->resource().device_type();
  } else {
    return DeviceType::kCPU;
  }
}

int64_t IDMgr::NewTaskId(int64_t machine_id, int64_t thrd_id) {
  int64_t machine_thrd_id = GetMachineThrdId(machine_id, thrd_id);
  CHECK_LT(thread_id2num_of_tasks_[machine_thrd_id],
           (static_cast<int64_t>(1) << task_id_bit_num_) - 1);
  return machine_thrd_id | (thread_id2num_of_tasks_[machine_thrd_id]++);
}

int64_t IDMgr::AllocatePersistenceThrdId(int64_t machine_id) {
  int64_t& offset = persistence_thrd_offset_[machine_id];
  int64_t ret = device_num_per_machine_ + offset;
  offset = (offset + 1) % JobDesc::Singleton()->PersistenceWorkerNum();
  return ret;
}
int64_t IDMgr::AllocateBoxingThrdId(int64_t machine_id) {
  int64_t offset = boxing_thrd_offset_[machine_id];
  int64_t ret = device_num_per_machine_
                + JobDesc::Singleton()->PersistenceWorkerNum() + offset;
  offset = (offset + 1) % JobDesc::Singleton()->BoxingWorkerNum();
  return ret;
}
int64_t IDMgr::CommNetThrdId() const {
  return device_num_per_machine_ + JobDesc::Singleton()->PersistenceWorkerNum()
         + JobDesc::Singleton()->BoxingWorkerNum();
}

DeviceType IDMgr::GetDeviceTypeFromActorId(int64_t actor_id) const {
  int64_t thrd_id = ThrdId4ActorId(actor_id);
  return GetDeviceTypeFromThrdId(thrd_id);
}

int64_t IDMgr::MachineId4ActorId(int64_t actor_id) const {
  return actor_id >> (63 - machine_id_bit_num_);
}

int64_t IDMgr::ThrdId4ActorId(int64_t actor_id) const {
  int64_t tmp = (actor_id << machine_id_bit_num_);
  tmp &= ~(static_cast<int64_t>(1) << 63);
  return tmp >> (machine_id_bit_num_ + task_id_bit_num_);
}

int64_t IDMgr::GetReservedWorkStreamId(int64_t machine_id, int64_t thrd_id,
                                       int64_t reserved_id) {
  CHECK_GE(reserved_id, static_cast<int64_t>(0));
  CHECK_LT(reserved_id, static_cast<int64_t>(1000));
  int64_t machine_thrd_id = GetMachineThrdId(machine_id, thrd_id);
  return machine_thrd_id | reserved_id;
}

int64_t IDMgr::NewWorkStreamId(int64_t machine_id, int64_t thrd_id) {
  int64_t machine_thrd_id = GetMachineThrdId(machine_id, thrd_id);
  int64_t& streams_num = thread_id2num_of_streams_[machine_thrd_id];
  if (streams_num < 1000) { streams_num = 1000; }
  CHECK_LT(streams_num, (static_cast<int64_t>(1) << task_id_bit_num_) - 1);
  return machine_thrd_id | (streams_num++);
}

IDMgr::IDMgr() {
  const Resource& resource = JobDesc::Singleton()->resource();
  machine_num_ = resource.machine_size();
  CHECK_LT(machine_num_, static_cast<int64_t>(1) << machine_id_bit_num_);
  device_num_per_machine_ = resource.device_num_per_machine();
  CHECK_LT(device_num_per_machine_,
           (static_cast<int64_t>(1) << thread_id_bit_num_) - 3);
  for (int64_t i = 0; i < machine_num_; ++i) {
    const std::string& machine_name = resource.machine(i).name();
    CHECK(machine_name2machine_id_.emplace(machine_name, i).second);
    CHECK(machine_id2machine_name_.emplace(i, machine_name).second);
  }
  regst_desc_id_count_ = 0;
  persistence_thrd_offset_.assign(machine_num_, 0);
  boxing_thrd_offset_.assign(machine_num_, 0);
}

int64_t IDMgr::GetMachineThrdId(int64_t machine_id, int64_t thrd_id) {
  int64_t machine_id64bit = machine_id << (63 - machine_id_bit_num_);
  int64_t thread_id64bit = thrd_id << task_id_bit_num_;
  int64_t machine_thread_id = machine_id64bit | thread_id64bit;
  return machine_thread_id;
}

}  // namespace oneflow
