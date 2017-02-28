#ifndef ONEFLOW_JOB_PARALLEL_DESCRIPTOR_H_
#define ONEFLOW_JOB_PARALLEL_DESCRIPTOR_H_

#include "common/util.h"
#include "common/id_map.h"

namespace oneflow {

class ParallelDescriptor {
 public:
  using ParallelPolicy = ParallelConf::Policy;
  static const ParallelPolicy kDataParallel = ParallelConf::DataParallel;
  static const ParallelPolicy kModelParallel = ParallelConf::ModelParallel;

  // DISALLOW_COPY_AND_MOVE(ParallelDescriptor);
  ParallelDescriptor() = default;
  ~ParallelDescriptor() = default;

  void Init(const ParallelConf& user_conf) {
    // TODO
  }
  
  const ParallelPolicy& policy() const { return policy_; }
  const std::unordered_set<MachineId>& machine_set() const {
    return machine_set_;
  }
  const std::unordered_set<DeviceId>& device_set() const {
    return device_set_;
  }

  bool operator == (const ParallelDescriptor& rhs) const {
    return policy_ == rhs.policy_
        && machine_set_ == rhs.machine_set_
        && device_set_ == rhs.device_set_;
  }
  bool operator != (const ParallelDescriptor& rhs) const {
    return !((*this) == rhs);
  }
  
 private:
  ParallelPolicy policy_;
  std::unordered_set<MachineId> machine_set_;
  std::unordered_set<DeviceId> device_set_;
};

} // namespace oneflow

#endif // ONEFLOW_JOB_PARALLEL_DESCRIPTOR_H_
