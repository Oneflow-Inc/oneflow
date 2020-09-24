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
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/id_manager.h"

namespace oneflow {

Maybe<void> ParseDeviceNameConf(const std::string& device_name, int64_t* mchn_id,
                                std::string* device_id_str) {
  size_t delimiter_pos = device_name.rfind(":");
  CHECK_NE_OR_RETURN(delimiter_pos, std::string::npos);
  *mchn_id = oneflow_cast<int64_t>(device_name.substr(0, delimiter_pos));
  *device_id_str = device_name.substr(delimiter_pos + 1);
  return Maybe<void>::Ok();
}

Maybe<OFRecord> ParseMachineAndDeviceIdList(const ParallelConf& parallel_conf) {
  ParallelDesc parallel_desc;
  JUST(parallel_desc.MaybeInit(parallel_conf));
  auto machine2device_list = std::make_shared<OFRecord>();
  auto* features = machine2device_list->mutable_feature();
  for (int64_t machine_id : parallel_desc.sorted_machine_ids()) {
    Int32List* device_id_list = (*features)[std::to_string(machine_id)].mutable_int32_list();
    for (int64_t device_id : parallel_desc.sorted_dev_phy_ids(machine_id)) {
      device_id_list->add_value(device_id);
    }
  }
  return machine2device_list;
}

ParallelDesc::ParallelDesc(const ParallelConf& user_conf) {
  CHECK_JUST(MaybeInit(user_conf));
  CHECK_JUST(CheckWithResourceDesc(*(Global<ResourceDesc, ForSession>::Get())));
}

Maybe<void> ParallelDesc::MaybeInit(const ParallelConf& user_conf) {
  parallel_conf_ = user_conf;
  HashSet<int64_t> machine_id_set;
  device_type_ = DeviceType::kInvalidDevice;
  const std::string& device_tag = parallel_conf_.device_tag();
  DeviceType device_type = JUST(DeviceType4DeviceTag(device_tag));
  CHECK_OR_RETURN(device_type_ == DeviceType::kInvalidDevice || device_type_ == device_type);
  device_type_ = device_type;
  for (const std::string& device_name : parallel_conf_.device_name()) {
    int64_t mchn_id;
    std::string device_id_str;
    JUST(ParseDeviceNameConf(device_name, &mchn_id, &device_id_str));
    machine_id_set.insert(mchn_id);
    if (machine_id_set.find(mchn_id) == machine_id_set.end()) {
      sorted_machine_ids_.push_back(mchn_id);
    }
    int64_t minus_pos = device_id_str.find("-");
    if (minus_pos == std::string::npos) {
      device_id_str = device_id_str + "-" + device_id_str;
      minus_pos = device_id_str.find("-");
    }
    int64_t min_id = oneflow_cast<int64_t>(device_id_str.substr(0, minus_pos));
    int64_t max_id = oneflow_cast<int64_t>(device_id_str.substr(minus_pos + 1));
    CHECK_LE_OR_RETURN(min_id, max_id);
    for (int64_t dev_phy_id = min_id; dev_phy_id <= max_id; ++dev_phy_id) {
      machine_id2sorted_dev_phy_ids_[mchn_id].push_back(dev_phy_id);
    }
  }
  ClearUp();
  JUST(SanityCheck());
  return Maybe<void>::Ok();
}

Maybe<void> ParallelDesc::GetParallelContext(ParallelContext* parallel_ctx, int64_t machine_id,
                                             int64_t device_id) const {
  parallel_ctx->set_parallel_num(parallel_num());
  const auto& machine_iter = machine_id2device_id2parallel_id_.find(machine_id);
  CHECK_OR_RETURN(machine_iter != machine_id2device_id2parallel_id_.end());
  const auto& device_iter = machine_iter->second.find(device_id);
  CHECK_OR_RETURN(device_iter != machine_iter->second.end());
  parallel_ctx->set_parallel_id(device_iter->second);
  return Maybe<void>::Ok();
}

bool ParallelDesc::Equals(const ParallelDesc& rhs) const {
  return device_type_ == rhs.device_type_ && sorted_machine_ids_ == rhs.sorted_machine_ids_
         && machine_id2sorted_dev_phy_ids_ == rhs.machine_id2sorted_dev_phy_ids_;
}

bool ParallelDesc::EqualsIgnoringDeviceType(const ParallelDesc& rhs) const {
  return sorted_machine_ids_ == rhs.sorted_machine_ids_
         && machine_id2sorted_dev_phy_ids_ == rhs.machine_id2sorted_dev_phy_ids_;
}

void ParallelDesc::ClearUp() {
  EraseIf<int64_t, std::vector<int64_t>>(
      &machine_id2sorted_dev_phy_ids_,
      [](HashMap<int64_t, std::vector<int64_t>>::iterator it) { return it->second.empty(); });
  sorted_machine_ids_.clear();
  parallel_num_ = 0;
  for (auto& pair : machine_id2sorted_dev_phy_ids_) {
    sorted_machine_ids_.push_back(pair.first);
    SortAndRemoveDuplication(&(pair.second));
    parallel_num_ += pair.second.size();
  }
  SortAndRemoveDuplication(&sorted_machine_ids_);
  int64_t parallel_id = 0;
  for (int64_t machine_id : sorted_machine_ids_) {
    for (int64_t device_id : machine_id2sorted_dev_phy_ids_.at(machine_id)) {
      parallel_id2machine_id_[parallel_id] = machine_id;
      parallel_id2device_id_[parallel_id] = device_id;
      machine_id2device_id2parallel_id_[machine_id][device_id] = parallel_id;
      parallel_id += 1;
    }
  }
}

void ParallelDesc::set_device_type(DeviceType device_type) {
  if (device_type == device_type_) { return; }
  device_type_ = device_type;
  const char* tag = CHECK_JUST(DeviceTag4DeviceType(device_type));
  parallel_conf_.set_device_tag(tag);
}

Maybe<void> ParallelDesc::SanityCheck() {
  device_num_of_each_machine_ = -1;
  for (auto& pair : machine_id2sorted_dev_phy_ids_) {
    if (device_num_of_each_machine_ == -1) {
      device_num_of_each_machine_ = pair.second.size();
    } else {
      CHECK_EQ_OR_RETURN(device_num_of_each_machine_, pair.second.size());
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> ParallelDesc::CheckWithResourceDesc(const ResourceDesc& resource_desc) {
  if (device_type_ == DeviceType::kGPU) {
    for (auto& pair : machine_id2sorted_dev_phy_ids_) {
      for (int64_t dev_phy_id : pair.second) {
        CHECK_LT_OR_RETURN(dev_phy_id, resource_desc.GpuDeviceNum());
      }
    }
  }
  return Maybe<void>::Ok();
}

ParallelConf ParallelDesc::GetParallelIdOnlyParallelConf(int64_t parallel_id) const {
  ParallelConf parallel_conf;
  std::string machine_id = std::to_string(MachineIdForParallelId(parallel_id));
  std::string device_id = std::to_string(DeviceIdForParallelId(parallel_id));
  parallel_conf.set_device_tag(CHECK_JUST(DeviceTag4DeviceType(device_type())));
  parallel_conf.add_device_name(machine_id + ":" + device_id);
  return parallel_conf;
}

int64_t ParallelDesc::MachineIdForParallelId(int64_t parallel_id) const {
  return parallel_id2machine_id_.at(parallel_id);
}

int64_t ParallelDesc::DeviceIdForParallelId(int64_t parallel_id) const {
  return parallel_id2device_id_.at(parallel_id);
}

bool ParallelDesc::Containing(int64_t machine_id, int64_t device_id) const {
  const auto& machine_iter = machine_id2sorted_dev_phy_ids_.find(machine_id);
  if (machine_iter == machine_id2sorted_dev_phy_ids_.end()) { return false; }
  const auto& vec = machine_iter->second;
  return std::find(vec.begin(), vec.end(), device_id) != vec.end();
}

std::tuple<int32_t, int32_t> GetPartIdAndPartNumFromParallelCtx(
    const ParallelContext* parallel_ctx) {
  return std::make_tuple(parallel_ctx->parallel_id(), parallel_ctx->parallel_num());
}

ParallelConf GenParallelConfOfCpuZeroOnMaster() {
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  parallel_conf.add_device_name("0:0");
  return parallel_conf;
}

ParallelConf GenParallelConfOfCpuZeroOnAllMachines() {
  ParallelConf parallel_conf;
  parallel_conf.set_device_tag("cpu");
  FOR_RANGE(int64_t, i, 0, (Global<ResourceDesc, ForSession>::Get()->TotalMachineNum())) {
    parallel_conf.add_device_name(std::to_string(i) + ":0");
  }
  return parallel_conf;
}

}  // namespace oneflow
