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

  OF_SINGLETON(IDMgr);

  // Compile
  int64_t MachineID4MachineName(const std::string& machine_name) const;
  const std::string& MachineName4MachineId(int64_t machine_id) const;
  DeviceType GetDeviceTypeFromThrdId(int64_t thrd_id) const;
  int64_t NewTaskId(int64_t machine_id, int64_t thrd_id);

  int64_t GetCpuDeviceThrdId(int64_t dev_phy_id) const { return dev_phy_id; }
  int64_t GetGpuDeviceThrdId(int64_t dev_phy_id) const;

  int64_t GetGpuDevPhyIdFromThrdId(int64_t thrd_id) const;

  int64_t AllocatePersistenceThrdId(int64_t machine_id);
  int64_t AllocateBoxingThrdId(int64_t machine_id);
  int64_t CommNetThrdId() const;
  int64_t NewRegstDescId() { return regst_desc_id_count_++; }

  // Runtime
  DeviceType GetDeviceTypeFromActorId(int64_t actor_id) const;
  int64_t MachineId4ActorId(int64_t actor_id) const;
  int64_t ThrdId4ActorId(int64_t actor_id) const;

  // reserved_id: 0-999
  // for cpu:
  //   0: the only one work stream
  // for gpu:
  //   0: the copy cuda stream
  int64_t GetReservedWorkStreamId(int64_t machine_id, int64_t thrd_id,
                                  int64_t reserved_id);
  // start from: 1000
  int64_t NewWorkStreamId(int64_t machine_id, int64_t thrd_id);

 private:
  IDMgr();
  int64_t GetMachineThrdId(int64_t machine_id, int64_t thrd_id);

  int64_t cpu_device_num_;
  int64_t gpu_device_num_;
  int64_t xpu_device_num_;
  int64_t regst_desc_id_count_;
  HashMap<int64_t, int64_t> thread_id2num_of_tasks_;
  HashMap<int64_t, int64_t> thread_id2num_of_streams_;

  HashMap<std::string, int64_t> machine_name2machine_id_;
  HashMap<int64_t, std::string> machine_id2machine_name_;

  std::vector<int64_t> persistence_thrd_offset_;
  std::vector<int64_t> boxing_thrd_offset_;

  //  64 bit id design:
  //   sign | machine | thread | task
  //    1   |   16    |   8    |  39
  static const int64_t machine_id_bit_num_ = 16;
  static const int64_t thread_id_bit_num_ = 8;
  static const int64_t task_id_bit_num_ = 39;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_ID_MANAGER_H_
