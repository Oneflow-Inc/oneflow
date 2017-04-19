#ifndef ONEFLOW_JOB_ID_MANAGER_H_
#define ONEFLOW_JOB_ID_MANAGER_H_

#include "common/util.h"
#include "job/resource.pb.h"
#include "job/id_manager.pb.h"

namespace oneflow {

class IDMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IDMgr);
  ~IDMgr() = default;

  static IDMgr& Singleton() {
    static IDMgr obj;
    return obj;
  }

  void InitFromResource(const Resource& resource) {
    machine_num_ = resource.machines_size();
    device_num_per_machine_ = resource.device_num_per_machine();
  }

  void InitFromProto(const IDMgrProto&) {
    TODO();
  }

  IDMgrProto ToProto() {
    TODO();
  }

  // Compile
  uint64_t ThrdLocId4DevicePhyId(uint64_t device_phy_id) const { 
    return device_phy_id;
  }
  uint64_t DiskThrdLocId() const {
    return device_num_per_machine_;
  }
  uint64_t BoxingThrdLocId() const {
    return device_num_per_machine_ + 1;
  }
  uint64_t CommNetThrdLocId() const { 
    return device_num_per_machine_ + 2;
  }

  uint64_t NewTaskId(uint64_t machine_id, uint64_t thrd_local_id) const {
    uint64_t thrd_id = machine_id & thrd_local_id;
    return thrd_id & (thread_id2num_of_tasks_[thrd_id]++ << 28);
  }
  uint64_t NewRegstDescId(uint64_t producer_task_id) const {
    return producer_task_id & (task_id2num_of_register_desc_[producer_task_id]++ << 16);
  }

  // Runtime

 private:
  IDMgr() = default;
  int32_t machine_num_;
  uint64_t device_num_per_machine_;
  unordered_map<uint64_t, uint64_t> thread_id2num_of_tasks_;
  unordered_map<uint64_t, uint64_t> task_id2num_of_register_desc_;

};

} // namespace oneflow

#endif // ONEFLOW_JOB_ID_MANAGER_H_
