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
  ParallelPolicy policy() const { return parallel_conf_.policy(); }
  const std::vector<int64_t>& sorted_machine_ids() const { return sorted_machine_ids_; }
  const std::vector<int64_t>& sorted_dev_phy_ids(int64_t machine_id) const {
    return machine_id2sorted_dev_phy_ids_.at(machine_id);
  }
  int64_t parallel_num() const { return parallel_num_; }
  int64_t device_num_of_each_machine() const { return device_num_of_each_machine_; }
  std::string device_names() const { return device_names_; }
  const ParallelConf& parallel_conf() const { return parallel_conf_; }

  // Setters
  void set_policy(ParallelPolicy val) { parallel_conf_.set_policy(val); }

  bool EqualsIgnoringPolicy(const ParallelDesc& rhs) const;
  bool Equals(const ParallelDesc& rhs) const;
  bool operator==(const ParallelDesc& rhs) const { return Equals(rhs); }
  bool operator!=(const ParallelDesc& rhs) const { return !(*this == rhs); }
  bool Equals(const ParallelDesc* rhs) const { return Equals(*rhs); }
  int64_t MachineIdForParallelId(int64_t parallel_id) const;
  int64_t DeviceIdForParallelId(int64_t parallel_id) const;

 private:
  void ClearUp();
  void SanityCheck();

  DeviceType device_type_;
  ParallelConf parallel_conf_;
  std::vector<int64_t> sorted_machine_ids_;
  HashMap<int64_t, std::vector<int64_t>> machine_id2sorted_dev_phy_ids_;
  int64_t parallel_num_;
  int64_t device_num_of_each_machine_;
  std::string device_names_;
  HashMap<int64_t, int64_t> parallel_id2machine_id_;
  HashMap<int64_t, int64_t> parallel_id2device_id_;
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
