#ifndef ONEFLOW_CORE_JOB_PARALLEL_DESC_H_
#define ONEFLOW_CORE_JOB_PARALLEL_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/placement.pb.h"

namespace oneflow {

void ParseDeviceNameConf(const std::string& device_name, int64_t* mchn_id, std::string* device_tag,
                         std::string* device_id_str);

class ParallelDesc final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ParallelDesc);
  ParallelDesc() = delete;
  ~ParallelDesc() = default;

  ParallelDesc(const ParallelDesc&) = default;
  ParallelDesc(const ParallelConf& user_conf);

  // Getters
  DeviceType device_type() const { return device_type_; }
  ParallelPolicy policy() const { return policy_; }
  const std::vector<int64_t>& sorted_machine_ids() const { return sorted_machine_ids_; }
  const std::vector<int64_t>& sorted_dev_phy_ids(int64_t machine_id) const {
    return machine_id2sorted_dev_phy_ids_.at(machine_id);
  }
  int64_t parallel_num() const { return parallel_num_; }
  int64_t device_num_of_each_machine() const { return device_num_of_each_machine_; }

  // Setters
  void set_policy(ParallelPolicy val) { policy_ = val; }
  void set_device_type(DeviceType device_type) { device_type_ = device_type; }
  void RemoveNeedlessDevice(const std::string& op_name, int32_t max_device_num);
  void RemoveNeedlessDevice(int32_t max_device_num) { RemoveNeedlessDevice("", max_device_num); }
  void RandomSelectOneDeviceAndRemoveTheOthers();
  void UseCPUDevicesOnMaster();

  //
  bool Equal(const ParallelDesc& rhs) const;
  bool operator==(const ParallelDesc& rhs) const { return Equal(rhs); }
  bool Equal(const ParallelDesc* rhs) const { return Equal(*rhs); }

 private:
  void ClearUp();
  void SanityCheck();

  DeviceType device_type_;
  ParallelPolicy policy_;
  std::vector<int64_t> sorted_machine_ids_;
  HashMap<int64_t, std::vector<int64_t>> machine_id2sorted_dev_phy_ids_;
  int64_t parallel_num_;
  int64_t device_num_of_each_machine_;
};

std::tuple<int32_t, int32_t> GetPartIdAndPartNumFromParallelCtx(
    const ParallelContext* parallel_ctx);

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::ParallelDesc> {
  size_t operator()(const oneflow::ParallelDesc& pr) const {
    std::string str;
    for (int machine_id : pr.sorted_machine_ids()) {
      str += "::" + std::to_string(machine_id) + ":";
      for (int dev_id : pr.sorted_dev_phy_ids(machine_id)) { str += std::to_string(dev_id) + ","; }
    }
    return hash<std::string>()(str);
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_JOB_PARALLEL_DESC_H_
