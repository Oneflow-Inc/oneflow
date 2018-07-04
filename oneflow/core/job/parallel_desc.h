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
  DeviceType device_type() const { return device_type_; }
  ParallelPolicy policy() const { return policy_; }
  const std::vector<int64_t>& sorted_machine_ids() const { return sorted_machine_ids_; }
  const std::vector<int64_t>& sorted_dev_phy_ids(int64_t machine_id) const {
    return machine_id2sorted_dev_phy_ids_.at(machine_id);
  }
  int64_t parallel_num() const { return parallel_num_; }

  // Setters
  void set_policy(ParallelPolicy val) { policy_ = val; }
  void set_device_type(DeviceType device_type) { device_type_ = device_type; }
  void RemoveNeedlessDevice(const std::string& op_name, int32_t max_device_num);
  void RemoveNeedlessDevice(int32_t max_device_num) { RemoveNeedlessDevice("", max_device_num); }
  void RandomSelectOneDeviceAndRemoveTheOthers();

  //
  bool Equal(const ParallelDesc& rhs) const;
  bool Equal(const ParallelDesc* rhs) const { return Equal(*rhs); }

 private:
  void ClearUp();

  DeviceType device_type_;
  ParallelPolicy policy_;
  std::vector<int64_t> sorted_machine_ids_;
  HashMap<int64_t, std::vector<int64_t>> machine_id2sorted_dev_phy_ids_;
  int64_t parallel_num_;
};

std::tuple<int32_t, int32_t> GetPartIdAndPartNumFromParallelCtx(
    const ParallelContext* parallel_ctx);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_PARALLEL_DESC_H_
