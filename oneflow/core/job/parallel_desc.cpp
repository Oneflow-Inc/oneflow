#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void ParseDeviceNameConf(const std::string& device_name, int64_t* mchn_id, std::string* device_tag,
                         std::string* device_id_str) {
  size_t second_delimiter_pos = device_name.rfind(":");
  CHECK_NE(second_delimiter_pos, std::string::npos);
  size_t first_delimiter_pos = device_name.rfind(":", second_delimiter_pos - 1);
  CHECK_NE(first_delimiter_pos, std::string::npos);
  *mchn_id = oneflow_cast<int64_t>(device_name.substr(0, first_delimiter_pos));
  *device_tag =
      device_name.substr(first_delimiter_pos + 1, second_delimiter_pos - first_delimiter_pos - 1);
  *device_id_str = device_name.substr(second_delimiter_pos + 1);
}

ParallelDesc::ParallelDesc(const ParallelConf& user_conf) {
  policy_ = user_conf.policy();
  HashSet<int64_t> machine_id_set;
  device_type_ = DeviceType::kInvalidDevice;
  for (const std::string& device_name : user_conf.device_name()) {
    int64_t mchn_id;
    std::string device_tag;
    std::string device_id_str;
    ParseDeviceNameConf(device_name, &mchn_id, &device_tag, &device_id_str);
    machine_id_set.insert(mchn_id);
    if (device_tag == "cpu") {
      CHECK(device_type_ == DeviceType::kInvalidDevice || device_type_ == DeviceType::kCPU);
      device_type_ = DeviceType::kCPU;
    } else if (device_tag == "gpu") {
      CHECK(device_type_ == DeviceType::kInvalidDevice || device_type_ == DeviceType::kGPU);
      device_type_ = DeviceType::kGPU;
    } else {
      UNIMPLEMENTED();
    }
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
    CHECK_LE(min_id, max_id);
    for (int64_t dev_phy_id = min_id; dev_phy_id <= max_id; ++dev_phy_id) {
      if (device_type_ == DeviceType::kGPU) {
        CHECK_LT(dev_phy_id, Global<JobDesc>::Get()->GpuDeviceNum());
      }
      machine_id2sorted_dev_phy_ids_[mchn_id].push_back(dev_phy_id);
    }
  }
  ClearUp();
  SanityCheck();
}

bool ParallelDesc::Equal(const ParallelDesc& rhs) const {
  return device_type_ == rhs.device_type_ && policy_ == rhs.policy_
         && sorted_machine_ids_ == rhs.sorted_machine_ids_
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
}

void ParallelDesc::SanityCheck() {
  device_num_of_each_machine_ = -1;
  for (auto& pair : machine_id2sorted_dev_phy_ids_) {
    if (device_num_of_each_machine_ == -1) {
      device_num_of_each_machine_ = pair.second.size();
    } else {
      CHECK_EQ(device_num_of_each_machine_, pair.second.size());
    }
  }
}

std::tuple<int32_t, int32_t> GetPartIdAndPartNumFromParallelCtx(
    const ParallelContext* parallel_ctx) {
  if (parallel_ctx->policy() == kDataParallel) {
    return std::make_tuple(0, 1);
  } else if (parallel_ctx->policy() == kModelParallel) {
    return std::make_tuple(parallel_ctx->parallel_id(), parallel_ctx->parallel_num());
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace oneflow
