#ifndef ONEFLOW_CORE_JOB_PARALLEL_DESC_H_
#define ONEFLOW_CORE_JOB_PARALLEL_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/placement.pb.h"

namespace oneflow {

std::string DeviceTag4DeviceType(DeviceType device_type);
Maybe<DeviceType> DeviceType4DeviceTag(const std::string& device_tag);

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
  const std::vector<int64_t>& sorted_machine_ids() const { return sorted_machine_ids_; }
  const std::vector<int64_t>& sorted_dev_phy_ids(int64_t machine_id) const {
    return machine_id2sorted_dev_phy_ids_.at(machine_id);
  }
  int64_t parallel_num() const { return parallel_num_; }
  int64_t device_num_of_each_machine() const { return device_num_of_each_machine_; }
  const ParallelConf& parallel_conf() const { return parallel_conf_; }

  // Setters
  void set_device_type(DeviceType device_type);

  bool EqualsIgnoringDeviceType(const ParallelDesc& rhs) const;
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
  HashMap<int64_t, int64_t> parallel_id2machine_id_;
  HashMap<int64_t, int64_t> parallel_id2device_id_;
};

inline bool operator==(const ParallelConf& lhs, const ParallelConf& rhs) {
  return ParallelDesc(lhs) == ParallelDesc(rhs);
}

inline bool operator!=(const ParallelConf& lhs, const ParallelConf& rhs) {
  return ParallelDesc(lhs) != ParallelDesc(rhs);
}

std::tuple<int32_t, int32_t> GetPartIdAndPartNumFromParallelCtx(
    const ParallelContext* parallel_ctx);

ParallelConf GenParallelConfOfCpuZeroOnMaster();
ParallelConf GenParallelConfOfCpuZeroOnAllMachines();

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::ParallelDesc> {
  size_t operator()(const oneflow::ParallelDesc& pr) const {
    size_t ret = 0;
    int i = 0;
    int shift_roundtrip = (sizeof(size_t) / 2);
    for (int machine_id : pr.sorted_machine_ids()) {
      int shift = i++ % shift_roundtrip;
      ret ^= machine_id << shift_roundtrip << shift;
      ret ^= pr.sorted_dev_phy_ids(machine_id).size() << shift;
    }
    return hash<size_t>()(ret);
  }
};

template<>
struct hash<oneflow::ParallelConf> {
  size_t operator()(const oneflow::ParallelConf& parallel_conf) const {
    return std::hash<oneflow::ParallelDesc>()(oneflow::ParallelDesc(parallel_conf));
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_JOB_PARALLEL_DESC_H_
