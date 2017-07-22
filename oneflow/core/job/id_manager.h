#ifndef ONEFLOW_CORE_JOB_ID_MANAGER_H_
#define ONEFLOW_CORE_JOB_ID_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

class IDMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IDMgr);
  ~IDMgr() = default;

  OF_SINGLETON(IDMgr);

  void InitFromResource(const Resource& resource) {
    LOG(INFO) << "Init IDManager";
    Clear();
    machine_num_ = resource.machine_size();
    CHECK_LT(machine_num_, static_cast<int64_t>(1) << machine_id_bit_num_);
    device_num_per_machine_ = resource.device_num_per_machine();
    // reserve 3 number of device_id for persistence_, boxing_ and commnet_
    // ThrdLocId
    CHECK_LT(device_num_per_machine_,
             (static_cast<int64_t>(1) << device_id_bit_num_) - 3);
    for (int64_t i = 0; i < machine_num_; ++i) {
      const std::string& machine_name = resource.machine(i).name();
      CHECK(machine_name2machine_id_.emplace(machine_name, i).second);
      CHECK(machine_id2machine_name_.emplace(i, machine_name).second);
    }
    regst_desc_id_count_ = 0;
  }

  // Compile
  int64_t MachineID4MachineName(const std::string& machine_name) const {
    return machine_name2machine_id_.at(machine_name);
  }
  std::string MachineName4MachineId(int64_t machine_id) const {
    return machine_id2machine_name_.at(machine_id);
  }
  int64_t ThrdLocId4DevPhyId(int64_t device_phy_id) const {
    return device_phy_id;
  }
  int64_t DevPhyId4ThrdLocId(int64_t thrd_loc_id) const {
    CHECK_LT(thrd_loc_id, device_num_per_machine_);
    return thrd_loc_id;
  }
  int64_t PersistenceThrdLocId() const { return device_num_per_machine_; }
  int64_t BoxingThrdLocId() const { return device_num_per_machine_ + 1; }
  int64_t CommNetThrdLocId() const { return device_num_per_machine_ + 2; }

  int64_t NewTaskId(int64_t machine_id, int64_t thrd_local_id) {
    int64_t machine_id64bit = machine_id << (63 - machine_id_bit_num_);
    int64_t device_id64bit = thrd_local_id << task_id_bit_num_;
    int64_t thrd_id = machine_id64bit | device_id64bit;
    CHECK_LT(thread_id2num_of_tasks_[thrd_id],
             (static_cast<int64_t>(1) << task_id_bit_num_) - 1);
    return thrd_id | (thread_id2num_of_tasks_[thrd_id]++);
  }
  int64_t NewRegstDescId() { return regst_desc_id_count_++; }

  // Runtime
  int64_t ActorId4TaskId(int64_t task_id) { return task_id; }
  int64_t TaskId4ActorId(int64_t actor_id) { return actor_id; }
  int64_t MachineId4ActorId(int64_t actor_id) {
    return actor_id >> (63 - machine_id_bit_num_);
  }
  int64_t ThrdLocId4ActorId(int64_t actor_id) {
    int64_t tmp = (actor_id << machine_id_bit_num_);
    tmp &= ~(static_cast<int64_t>(1) << 63);
    return tmp >> (machine_id_bit_num_ + task_id_bit_num_);
  }

 private:
  IDMgr() = default;
  void Clear() {
    thread_id2num_of_tasks_.clear();
    machine_id2machine_name_.clear();
    machine_name2machine_id_.clear();
  }
  int32_t machine_num_;
  int64_t device_num_per_machine_;
  int64_t regst_desc_id_count_;
  HashMap<int64_t, int64_t> thread_id2num_of_tasks_;
  // machine_id is like {0, 1, 2, 3, ...}
  HashMap<std::string, int64_t> machine_name2machine_id_;
  HashMap<int64_t, std::string> machine_id2machine_name_;

  //  64 bit id design:
  //   sign | machine | device | task
  //    1   |   16    |   8    |  39
  static const int64_t machine_id_bit_num_ = 16;
  static const int64_t device_id_bit_num_ = 8;
  static const int64_t task_id_bit_num_ = 39;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_ID_MANAGER_H_
