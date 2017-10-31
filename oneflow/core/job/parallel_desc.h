#ifndef ONEFLOW_CORE_JOB_PARALLEL_DESC_H_
#define ONEFLOW_CORE_JOB_PARALLEL_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/placement.pb.h"

namespace oneflow {

class ParallelDesc {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ParallelDesc);
  ParallelDesc() = delete;
  ~ParallelDesc() = default;

  ParallelDesc(const ParallelConf& user_conf);

  // Getters
  ParallelPolicy policy() const { return policy_; }
  const std::vector<int64_t>& sorted_machine_ids() const {
    return sorted_machine_ids_;
  }
  const std::vector<int64_t>& sorted_dev_phy_ids(int64_t machine_id) const {
    return machine_id2sorted_dev_phy_ids_.at(machine_id);
  }
  int64_t parallel_num() const { return parallel_num_; }

  // Setters
  void set_policy(ParallelPolicy val) { policy_ = val; }
  void RemoveNeedlessDevice(int32_t max_device_num);
  void RemoveInvalidDevice();

  //
  bool Equal(const ParallelDesc& rhs) const;
  bool Equal(const ParallelDesc* rhs) const { return Equal(*rhs); }

 private:
  void ClearUp();

  ParallelPolicy policy_;
  std::vector<int64_t> sorted_machine_ids_;
  HashMap<int64_t, std::vector<int64_t>> machine_id2sorted_dev_phy_ids_;
  int64_t parallel_num_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_PARALLEL_DESC_H_
