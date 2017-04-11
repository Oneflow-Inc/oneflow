#ifndef ONEFLOW_JOB_PARALLEL_DESC_H_
#define ONEFLOW_JOB_PARALLEL_DESC_H_

#include "common/util.h"
#include "job/id_manager.h"
#include "job/strategy.pb.h"

namespace oneflow {

class ParallelDesc {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ParallelDesc);
  ParallelDesc() = delete;
  ~ParallelDesc() = default;

  ParallelDesc(const ParallelConf& user_conf);
  
  // Getters
  const ParallelPolicy& policy() const { return policy_; }
  const DeviceType& device_type() const { return device_type_; } 
  const std::vector<MachineId>& sorted_machines() const {
    return sorted_machine_vec_;
  }
  const std::vector<DevicePhyId>& sorted_devices_on_machine(MachineId machine_id) const {
    return sorted_devices_on_machine_.at(machine_id);
  }

  //
  ParallelPolicy& mut_policy() { return policy_; }
  bool operator == (const ParallelDesc& rhs) const {
    TODO();
  }
  bool operator != (const ParallelDesc& rhs) const {
    return !((*this) == rhs);
  }
  
 private:
  ParallelPolicy policy_;
  DeviceType device_type_;
  std::vector<MachineId> sorted_machine_vec_;
  HashMap<MachineId, std::vector<DevicePhyId>> sorted_devices_on_machine_;

};

} // namespace oneflow

#endif // ONEFLOW_JOB_PARALLEL_DESC_H_
