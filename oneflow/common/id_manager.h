#ifndef ONEFLOW_COMMON_ID_MANAGER_H_
#define ONEFLOW_COMMON_ID_MANAGER_H_

#include "oneflow/common/util.h"
#include "oneflow/conf/resource.pb.h"

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
    CHECK_LT(machine_num_, 1 << machine_id_bit_num_);
    device_num_per_machine_ = resource.device_num_per_machine();
    // reserve 3 number of device_id for persistence_, boxing_ and commnet_ ThrdLocId
    CHECK_LT(device_num_per_machine_, (1 << device_id_bit_num_) - 3);
    for (uint64_t i = 0; i < machine_num_; ++i) {
      const std::string& machine_name = resource.machine(i).name();
      CHECK(machine_name2machine_id_.emplace(machine_name, i).second);
      CHECK(machine_id2machine_name_.emplace(i, machine_name).second);
    }
  }

  // Compile
  uint64_t MachineID4MachineName(const std::string& machine_name) const {
    return machine_name2machine_id_.at(machine_name);
  }
  std::string MachineName4MachineId(uint64_t machine_id) const {
    return machine_id2machine_name_.at(machine_id);
  }
  uint64_t ThrdLocId4DevPhyId(uint64_t device_phy_id) const {
    return device_phy_id;
  }
  uint64_t DevPhyId4ThrdLocId(uint64_t thrd_loc_id) const {
    CHECK_LT(thrd_loc_id, device_num_per_machine_);
    return thrd_loc_id;
  }
  uint64_t PersistenceThrdLocId() const { return device_num_per_machine_; }
  uint64_t BoxingThrdLocId() const { return device_num_per_machine_ + 1; }
  uint64_t CommNetThrdLocId() const { return device_num_per_machine_ + 2; }

  uint64_t NewTaskId(uint64_t machine_id, uint64_t thrd_local_id) {
    uint64_t machine_id64bit = machine_id << (64 - machine_id_bit_num_);
    uint64_t device_id64bit = thrd_local_id <<
        (task_id_bit_num_ + regst_desc_id_bit_num_ + register_id_bit_num_);
    uint64_t thrd_id = machine_id64bit | device_id64bit;
    CHECK_LT(thread_id2num_of_tasks_[thrd_id], (1 << task_id_bit_num_) - 1);
    return thrd_id | ((thread_id2num_of_tasks_[thrd_id]++) << 
        (regst_desc_id_bit_num_ + register_id_bit_num_));
  }
  uint64_t NewRegstDescId(uint64_t producer_task_id) {
    CHECK_LT(task_id2num_of_register_desc_[producer_task_id], 
             (1 << regst_desc_id_bit_num_) - 1);
    return producer_task_id |
        ((task_id2num_of_register_desc_[producer_task_id]++) << register_id_bit_num_);
  }

  // Runtime
  uint64_t GetActorIdFromTaskId(uint64_t task_id) {
    return task_id;
  }
  uint64_t NewRegstId(uint64_t regst_desc_id) {
    CHECK_LT(regst_desc_id2num_of_register_[regst_desc_id],
             (1 << register_id_bit_num_) - 1);
    return regst_desc_id | regst_desc_id2num_of_register_[regst_desc_id]++;
  }
  uint64_t MachineId4ActorId(uint64_t actor_id) {
    return actor_id >> (64 - machine_id_bit_num_);
  }
  uint64_t ThrdLocId4ActorId(uint64_t actor_id) {
    return (actor_id << machine_id_bit_num_) >> (64 - device_id_bit_num_);
  }

 private:
  IDMgr() = default;
  int32_t machine_num_;
  uint64_t device_num_per_machine_;
  HashMap<uint64_t, uint64_t> thread_id2num_of_tasks_;
  HashMap<uint64_t, uint64_t> task_id2num_of_register_desc_;
  // machine_id is like {0, 1, 2, 3, ...}
  HashMap<std::string, uint64_t> machine_name2machine_id_;
  HashMap<uint64_t, std::string> machine_id2machine_name_;
  HashMap<uint64_t, uint64_t> regst_desc_id2num_of_register_;

  //  64 bit id design:
  //    machine | device | task | regst_desc | regst
  //      12    |   8    |  16  |     12     |   16
  static const uint64_t machine_id_bit_num_ = 12;
  static const uint64_t device_id_bit_num_ = 8;
  static const uint64_t task_id_bit_num_ = 16;
  static const uint64_t regst_desc_id_bit_num_ = 12;
  static const uint64_t register_id_bit_num_ = 16;

};

}  // namespace oneflow

#endif  // ONEFLOW_COMMON_ID_MANAGER_H_
