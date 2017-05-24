#ifndef ONEFLOW_COMMON_ID_MANAGER_H_
#define ONEFLOW_COMMON_ID_MANAGER_H_

#include "common/util.h"
#include "conf/resource.pb.h"

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
    LOG(INFO) << "Init IDManager...";
    machine_num_ = resource.machine_size();
    device_num_per_machine_ = resource.device_num_per_machine();
    for (uint64_t i = 0; i < machine_num_; ++i) {
      const std::string& machine_name = resource.machine(i).name();
      CHECK(machine_name2machine_id_.emplace(machine_name, i).second);
    }
  }

  // Compile
  uint64_t MachineID4MachineName(const std::string& machine_name) const {
    return machine_name2machine_id_.at(machine_name);
  }
  uint64_t ThrdLocId4DevPhyId(uint64_t device_phy_id) const {
    return device_phy_id;
  }
  uint64_t DevPhyId4ThrdLocId(uint64_t thrd_loc_id) const {
    CHECK_LT(thrd_loc_id, device_num_per_machine_);
    return thrd_loc_id;
  }
  uint64_t DiskThrdLocId() const { return device_num_per_machine_; }
  uint64_t BoxingThrdLocId() const { return device_num_per_machine_ + 1; }
  uint64_t CommNetThrdLocId() const { return device_num_per_machine_ + 2; }

  uint64_t NewTaskId(uint64_t machine_id, uint64_t thrd_local_id) {
    uint64_t thrd_id = (machine_id | (thrd_local_id << 48));
    return thrd_id | ((thread_id2num_of_tasks_[thrd_id]++) << 28);
  }
  uint64_t NewRegstDescId(uint64_t producer_task_id) {
    return producer_task_id |
           ((task_id2num_of_register_desc_[producer_task_id]++) << 16);
  }

  // Runtime
  uint64_t GetActorIdFromTaskId(uint64_t task_id) {
    return task_id;
  }
  uint64_t NewRegstId(uint64_t regst_desc_id) {
    TODO();
  }
  uint64_t MachineId4ActorId(uint64_t actor_id) {
    TODO();
  }
  uint64_t ThrdLocId4ActorId(uint64_t actor_id) {
    TODO();
  }

 private:
  IDMgr() = default;
  int32_t machine_num_;
  uint64_t device_num_per_machine_;
  HashMap<uint64_t, uint64_t> thread_id2num_of_tasks_;
  HashMap<uint64_t, uint64_t> task_id2num_of_register_desc_;
  HashMap<std::string, uint64_t> machine_name2machine_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_COMMON_ID_MANAGER_H_
