#ifndef ONEFLOW_JOB_PARALLEL_DESC_H_
#define ONEFLOW_JOB_PARALLEL_DESC_H_

#include "common/util.h"
#include "job/id_manager.h"
#include "job/strategy.pb.h"
#include "job/job_desc.h"
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
  const std::vector<int64_t>& sorted_machine_ids() const {
    return sorted_machine_ids_;
  }
  const std::vector<int64_t>& sorted_device_phy_ids(int64_t machine_id) const {
    // If this is used to describe the disk
    // the return shouble be empty
    return machine_id2sorted_device_phy_ids_.at(machine_id);
  }

  //
  ParallelPolicy& mut_policy() { return policy_; }
  bool operator == (const ParallelDesc& rhs) const {
	  return policy_ == rhs.policy_ 
		  && device_type_ == rhs.device_type_ 
		  && sorted_machine_ids_ == rhs.sorted_machine_ids_ 
		  && machine_id2sorted_device_phy_ids_ == rhs.machine_id2sorted_device_phy_ids_;
  }
  bool operator != (const ParallelDesc& rhs) const {
    return !((*this) == rhs);
  }
  
 private:
  ParallelPolicy policy_;
  DeviceType device_type_;
  std::vector<int64_t> sorted_machine_ids_;
  HashMap<int64_t, std::vector<int64_t>> machine_id2sorted_device_phy_ids_;

};

} // namespace oneflow

#endif // ONEFLOW_JOB_PARALLEL_DESC_H_
