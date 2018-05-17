#include "oneflow/core/job/id_manager.h"

namespace oneflow {

int64_t IDMgr::MachineID4MachineName(const std::string& machine_name) const {
  auto it = machine_name2machine_id_.find(machine_name);
  CHECK(it != machine_name2machine_id_.end()) << "Undefined machine name: " << machine_name;
  return it->second;
}

int64_t IDMgr::NewLocalWorkStreamId(int64_t machine_id, int64_t thrd_id) {
  int64_t local_work_stream_id = -1;
  if (GetDeviceTypeFromThrdId(thrd_id) == DeviceType::kCPU) {
    local_work_stream_id = 0;
  } else if (GetDeviceTypeFromThrdId(thrd_id) == DeviceType::kGPU) {
    // start from: 100
    local_work_stream_id =
        100 + (thread_id2num_of_streams_[GetMachineThrdId(machine_id, thrd_id)]++);
  } else {
    UNIMPLEMENTED();
  }
  CHECK_GE(local_work_stream_id, 0);
  return local_work_stream_id;
}

int64_t IDMgr::NewTaskId(int64_t machine_id, int64_t thrd_id, int64_t local_work_stream_id) {
  int64_t machine_thrd_id = GetMachineThrdId(machine_id, thrd_id);
  CHECK_LT(thread_id2num_of_tasks_[machine_thrd_id],
           (static_cast<int64_t>(1) << task_id_bit_num_) - 1);
  CHECK_LT(local_work_stream_id, static_cast<int64_t>(1) << local_work_stream_id_bit_num_);
  return machine_thrd_id | (local_work_stream_id << task_id_bit_num_)
         | (thread_id2num_of_tasks_[machine_thrd_id]++);
}

DeviceType IDMgr::GetDeviceTypeFromThrdId(int64_t thrd_id) const {
  if (thrd_id < gpu_device_num_) {
    return DeviceType::kGPU;
  } else {
    return DeviceType::kCPU;
  }
}

int64_t IDMgr::GetGpuDevPhyIdFromThrdId(int64_t thrd_id) const {
  CHECK_LT(thrd_id, gpu_device_num_);
  return thrd_id;
}

int64_t IDMgr::GetMemZoneIdFromThrdId(int64_t thrd_id) const {
  return std::min(thrd_id, gpu_device_num_);
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
  return tmp >> (63 - thread_id_bit_num_);
}

int64_t IDMgr::GlobalWorkStreamId4ActorId(int64_t actor_id) const {
  return (actor_id >> task_id_bit_num_) << task_id_bit_num_;
}

int64_t IDMgr::LocalWorkStreamId4TaskId(int64_t task_id) const {
  return LocalWorkStreamId4ActorId(task_id);
}

int64_t IDMgr::LocalWorkStreamId4ActorId(int64_t actor_id) const {
  int64_t tmp = (actor_id << (machine_id_bit_num_ + thread_id_bit_num_));
  tmp &= ~(static_cast<int64_t>(1) << 63);
  return tmp >> (63 - local_work_stream_id_bit_num_);
}

IDMgr::IDMgr() {
  const Resource& resource = Global<JobDesc>::Get()->resource();
  int64_t machine_num = resource.machine_size();
  CHECK_LT(machine_num, static_cast<int64_t>(1) << machine_id_bit_num_);
  gpu_device_num_ = resource.gpu_device_num();
  cpu_device_num_ = resource.cpu_device_num();
  CHECK_LT(gpu_device_num_ + cpu_device_num_, (static_cast<int64_t>(1) << thread_id_bit_num_) - 3);
  for (int64_t i = 0; i < machine_num; ++i) {
    const std::string& machine_name = resource.machine(i).name();
    CHECK(machine_name2machine_id_.emplace(machine_name, i).second);
    CHECK(machine_id2machine_name_.emplace(i, machine_name).second);
  }
  regst_desc_id_count_ = 0;
}

int64_t IDMgr::GetMachineThrdId(int64_t machine_id, int64_t thrd_id) {
  int64_t machine_id64bit = machine_id << (63 - machine_id_bit_num_);
  int64_t thread_id64bit = thrd_id << (local_work_stream_id_bit_num_ + task_id_bit_num_);
  int64_t machine_thread_id = machine_id64bit | thread_id64bit;
  return machine_thread_id;
}

}  // namespace oneflow
