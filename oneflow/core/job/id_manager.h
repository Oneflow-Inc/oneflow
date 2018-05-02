#ifndef ONEFLOW_CORE_JOB_ID_MANAGER_H_
#define ONEFLOW_CORE_JOB_ID_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

class IDMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IDMgr);
  ~IDMgr() = default;

  // machine_name <-> machine_id
  int64_t MachineID4MachineName(const std::string& machine_name) const;
  const std::string& MachineName4MachineId(int64_t machine_id) const {
    return machine_id2machine_name_.at(machine_id);
  }

  // Get ThrdId, TaskId, RegstDescId
  int64_t GetGpuDeviceThrdId(int64_t dev_phy_id) const { return dev_phy_id; }
  int64_t GetCpuDeviceThrdId(int64_t dev_phy_id) const { return gpu_device_num_ + dev_phy_id; }
  int64_t GetPersistenceThrdId(int64_t offset) const {
    return gpu_device_num_ + cpu_device_num_ + offset;
  }
  int64_t CommNetThrdId() const {
    return gpu_device_num_ + cpu_device_num_ + Global<JobDesc>::Get()->PersistenceWorkerNum();
  }
  int64_t NewTaskId(int64_t machine_id, int64_t thrd_id);
  int64_t NewRegstDescId() { return regst_desc_id_count_++; }

  // Get MemZoneId
  int64_t CpuMemZoneId() const { return Global<JobDesc>::Get()->GpuDeviceNum(); }
  int64_t GpuMemZoneId(int64_t dev_phy_id) { return dev_phy_id; }

  // GetFromThrdId
  DeviceType GetDeviceTypeFromThrdId(int64_t thrd_id) const;
  int64_t GetGpuDevPhyIdFromThrdId(int64_t thrd_id) const;
  int64_t GetMemZoneIdFromThrdId(int64_t thrd_id) const;

  int64_t GetThrdIdFromGpuMemZoneId(int64_t mem_zone_id) {
    CHECK_NE(mem_zone_id, CpuMemZoneId());
    return mem_zone_id;
  }

  // Runtime
  DeviceType GetDeviceTypeFromActorId(int64_t actor_id) const;
  int64_t MachineId4ActorId(int64_t actor_id) const;
  int64_t ThrdId4ActorId(int64_t actor_id) const;

  // reserved_id: 0-999
  // for cpu:
  //   0: the actor thread
  // for gpu:
  //   0: the copy h2d cuda stream
  //   1: the copy d2h cuda stream
  int64_t GetReservedWorkStreamId(int64_t machine_id, int64_t thrd_id, int64_t reserved_id);
  // start from: 1000
  int64_t NewWorkStreamId(int64_t machine_id, int64_t thrd_id);

 private:
  friend class Global<IDMgr>;
  IDMgr();
  int64_t GetMachineThrdId(int64_t machine_id, int64_t thrd_id);

  int64_t gpu_device_num_;
  int64_t cpu_device_num_;
  int64_t regst_desc_id_count_;
  HashMap<int64_t, int64_t> thread_id2num_of_tasks_;
  HashMap<int64_t, int64_t> thread_id2num_of_streams_;

  HashMap<std::string, int64_t> machine_name2machine_id_;
  HashMap<int64_t, std::string> machine_id2machine_name_;

  //  64 bit id design:
  //   sign | machine | thread | task
  //    1   |   16    |   8    |  39
  static const int64_t machine_id_bit_num_ = 16;
  static const int64_t thread_id_bit_num_ = 8;
  static const int64_t task_id_bit_num_ = 39;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_ID_MANAGER_H_
