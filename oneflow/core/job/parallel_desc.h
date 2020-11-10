/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_JOB_PARALLEL_DESC_H_
#define ONEFLOW_CORE_JOB_PARALLEL_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/framework/to_string.h"

namespace oneflow {

class ResourceDesc;

Maybe<OFRecord> ParseMachineAndDeviceIdList(const ParallelConf& parallel_conf);

Maybe<void> ParseDeviceNameConf(const std::string& device_name, int64_t* mchn_id,
                                std::string* device_id_str);

class ParallelContext;

// Describe the information of devices used for parallel including device type, number of device,
// etc.
class ParallelDesc final {
 public:
  ~ParallelDesc() = default;

  ParallelDesc(const ParallelDesc&) = default;
  ParallelDesc(const ParallelConf& user_conf);
  Maybe<void> MaybeInit(const ParallelConf& user_conf);

  // Getters
  DeviceType device_type() const { return device_type_; }
  bool HasMachineId(int64_t machine_id) const {
    return machine_id2sorted_dev_phy_ids_.find(machine_id) != machine_id2sorted_dev_phy_ids_.end();
  }
  const std::vector<int64_t>& sorted_machine_ids() const { return sorted_machine_ids_; }
  const std::vector<int64_t>& sorted_dev_phy_ids(int64_t machine_id) const {
    return machine_id2sorted_dev_phy_ids_.at(machine_id);
  }
  int64_t parallel_num() const { return parallel_num_; }
  int64_t device_num_of_each_machine() const { return device_num_of_each_machine_; }
  const ParallelConf& parallel_conf() const { return parallel_conf_; }

  Maybe<void> GetParallelContext(ParallelContext* parallel_ctx, int64_t machine_id,
                                 int64_t device_id) const;

  // Setters
  void set_device_type(DeviceType device_type);

  ParallelConf GetParallelIdOnlyParallelConf(int64_t parallel_id) const;

  bool EqualsIgnoringDeviceType(const ParallelDesc& rhs) const;
  bool Equals(const ParallelDesc& rhs) const;
  bool operator==(const ParallelDesc& rhs) const { return Equals(rhs); }
  bool operator!=(const ParallelDesc& rhs) const { return !(*this == rhs); }
  bool Equals(const ParallelDesc* rhs) const { return Equals(*rhs); }
  Maybe<int64_t> MachineId4ParallelId(int64_t parallel_id) const;
  Maybe<int64_t> DeviceId4ParallelId(int64_t parallel_id) const;
  Maybe<int64_t> ParallelId4MachineDeviceId(int64_t machine_id, int64_t device_id) const;
  bool Containing(int64_t machine_id, int64_t device_id) const;
  bool ContainingMachineId(int64_t machine_id) const;

 private:
  friend Maybe<OFRecord> ParseMachineAndDeviceIdList(const ParallelConf& parallel_conf);
  ParallelDesc() = default;
  void ClearUp();
  Maybe<void> SanityCheck();
  Maybe<void> CheckWithResourceDesc(const ResourceDesc& resource_desc);

  DeviceType device_type_;
  ParallelConf parallel_conf_;
  std::vector<int64_t> sorted_machine_ids_;
  HashMap<int64_t, std::vector<int64_t>> machine_id2sorted_dev_phy_ids_;
  int64_t parallel_num_;
  int64_t device_num_of_each_machine_;
  // There is a one-to-one map between parallel_id and (machine_id, device_id)
  HashMap<int64_t, int64_t> parallel_id2machine_id_;
  HashMap<int64_t, int64_t> parallel_id2device_id_;
  HashMap<int64_t, HashMap<int64_t, int64_t>> machine_id2device_id2parallel_id_;
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
