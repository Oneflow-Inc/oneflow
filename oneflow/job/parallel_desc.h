#ifndef ONEFLOW_JOB_PARALLEL_DESC_H_
#define ONEFLOW_JOB_PARALLEL_DESC_H_

#include <unordered_map>
#include "common/util.h"
#include "common/id_map.h"

namespace oneflow {

class ParallelDesc {
 public:
  using ParallelPolicy = ParallelConf::Policy;
  static const ParallelPolicy kDataParallel = ParallelConf::DataParallel;
  static const ParallelPolicy kModelParallel = ParallelConf::ModelParallel;
  enum class Engine {
    kHost,
    kDevice
  };

  // DISALLOW_COPY_AND_MOVE(ParallelDesc);
  ParallelDesc() = default;
  ~ParallelDesc() = default;

  void Init(const ParallelConf& user_conf) {
    // TODO
  }
  
  const ParallelPolicy& policy() const { return policy_; }
  const Engine& engine() const { return engine_; } 
  const std::vector<MachineId>& machines() const {
    return machine_vec_;
  }
  const std::vector<DeviceId>& devices() const {
    return device_vec_;
  }
  const std::vector<DeviceId>& devices_on_machine(MachineId machine_id) const {
    return devices_on_machine_.at(machine_id);
  }

  bool operator == (const ParallelDesc& rhs) const {
    // TODO
  }
  bool operator != (const ParallelDesc& rhs) const {
    return !((*this) == rhs);
  }
  
 private:
  ParallelPolicy policy_;
  Engine engine_;
  // It is redundant for easy-using
  std::vector<MachineId> machine_vec_;
  std::vector<DeviceId> device_vec_;
  std::unordered_map<MachineId, std::vector<DeviceId>> devices_on_machine_;

};

} // namespace oneflow

#endif // ONEFLOW_JOB_PARALLEL_DESC_H_
