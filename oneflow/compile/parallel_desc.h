#ifndef ONEFLOW_COMPILE_PARALLEL_DESC_H_
#define ONEFLOW_COMPILE_PARALLEL_DESC_H_

#include "common/util.h"
#include "common/id_manager.h"
#include "conf/strategy.pb.h"
#include "common/job_desc.h"
#include <exception>

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
  const std::vector<uint64_t>& sorted_machine_ids() const {
    return sorted_machine_ids_;
  }
  const std::vector<uint64_t>& sorted_device_phy_ids(int64_t machine_id) const {
    // If this is used to describe the persistence
    // the return should be empty
    return machine_id2sorted_device_phy_ids_.at(machine_id);
  }
  uint64_t parallel_num() const { return parallel_num_; }

  //
  ParallelPolicy& mut_policy() { return policy_; }
  bool Equal(const ParallelDesc& rhs) const {
	  return policy_ == rhs.policy_ 
		  && device_type_ == rhs.device_type_ 
		  && sorted_machine_ids_ == rhs.sorted_machine_ids_ 
		  && machine_id2sorted_device_phy_ids_ == rhs.machine_id2sorted_device_phy_ids_;
  }
  bool Equal(const ParallelDesc* rhs) const {
    return Equal(*rhs);
  }

  std::string VisualStr() const;
  
 private:
  ParallelPolicy policy_;
  DeviceType device_type_;
  std::vector<uint64_t> sorted_machine_ids_;
  HashMap<uint64_t, std::vector<uint64_t>> machine_id2sorted_device_phy_ids_;
  uint64_t parallel_num_;
};

std::string GetMachineNameFromDeviceName(const std::string& device_name);

} // namespace oneflow

#endif // ONEFLOW_COMPILE_PARALLEL_DESC_H_
