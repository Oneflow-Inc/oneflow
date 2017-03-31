#ifndef ONEFLOW_JOB_PARALLEL_DESC_H_
#define ONEFLOW_JOB_PARALLEL_DESC_H_

#include <unordered_map>
#include "common/util.h"
#include "common/id_manager.h"
#include "job/strategy.pb.h"

namespace oneflow {

class ParallelDesc {
 public:
  using Policy = ParallelConf::Policy;
  enum class Engine {
    kHost,
    kDevice
  };

  // OF_DISALLOW_COPY_AND_MOVE(ParallelDesc);
  ParallelDesc() = delete;
  ~ParallelDesc() = default;

  ParallelDesc(const ParallelConf& user_conf) {
    TODO();
  }
  
  const Policy& policy() const { return policy_; }
  const Engine& engine() const { return engine_; } 
  const std::vector<MachineId>& sorted_machines() const {
    return sorted_machine_vec_;
  }
  const std::vector<DeviceGlobalId>& sorted_devices() const {
    return sorted_device_vec_;
  }
  const std::vector<DevicePhysicalId>& sorted_devices_on_machine(MachineId machine_id) const {
    return sorted_devices_on_machine_.at(machine_id);
  }

  bool operator == (const ParallelDesc& rhs) const {
    TODO();
  }
  bool operator != (const ParallelDesc& rhs) const {
    return !((*this) == rhs);
  }
  
 private:
  Policy policy_;
  Engine engine_;
  std::vector<MachineId> sorted_machine_vec_;
  std::vector<DeviceGlobalId> sorted_device_vec_;
  std::unordered_map<MachineId, std::vector<DevicePhysicalId>> sorted_devices_on_machine_;

};

static const ParallelDesc::Policy kDataParallel = ParallelConf::DataParallel;
static const ParallelDesc::Policy kModelParallel = ParallelConf::ModelParallel;

} // namespace oneflow

#endif // ONEFLOW_JOB_PARALLEL_DESC_H_
