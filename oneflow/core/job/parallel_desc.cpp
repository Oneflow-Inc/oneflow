#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace {

void ParseDeviceNameConf(const std::string& device_name, std::string* mchn_name,
                         std::string* device_id_str) {
  size_t delimiter_pos = device_name.rfind(":");
  CHECK_NE(delimiter_pos, std::string::npos);
  *mchn_name = device_name.substr(0, delimiter_pos);
  *device_id_str = device_name.substr(delimiter_pos + 1);
}

}  // namespace

ParallelDesc::ParallelDesc(const ParallelConf& user_conf) {
  policy_ = user_conf.policy();
  for (const std::string& device_name : user_conf.device_name()) {
    std::string mchn_name;
    std::string device_id_str;
    ParseDeviceNameConf(device_name, &mchn_name, &device_id_str);
    int64_t machine_id = IDMgr::Singleton()->MachineID4MachineName(mchn_name);
    sorted_machine_ids_.push_back(machine_id);
    int64_t minus_pos = device_id_str.rfind("-");
    if (minus_pos == std::string::npos) {
      int64_t dev_phy_id = oneflow_cast<int64_t>(device_id_str);
      machine_id2sorted_dev_phy_ids_[machine_id] = {dev_phy_id};
      continue;
    }
    int64_t min_id = oneflow_cast<int64_t>(device_id_str.substr(0, minus_pos));
    int64_t max_id = oneflow_cast<int64_t>(device_id_str.substr(minus_pos + 1));
    CHECK_LE(min_id, max_id);
    for (int64_t dev_phy_id = min_id; dev_phy_id <= max_id; ++dev_phy_id) {
      machine_id2sorted_dev_phy_ids_[machine_id].push_back(dev_phy_id);
    }
  }
  ClearUp();
}

void ParallelDesc::RemoveNeedlessDevice(const std::string& op_name,
                                        int32_t max_device_num) {
  if (max_device_num >= parallel_num_) { return; }
  LOG_IF(WARNING, op_name != "")
      << "parallel_num of " << op_name << " is greater than max_device_num "
      << max_device_num;
  int32_t device_cnt = 0;
  int64_t max_machine_id = -1;
  for (int64_t machine_id : sorted_machine_ids_) {
    auto it = machine_id2sorted_dev_phy_ids_.find(machine_id);
    int32_t cur_device_num = it->second.size();
    int32_t cur_device_max_num = max_device_num - device_cnt;
    if (cur_device_num > cur_device_max_num) {
      it->second.erase(it->second.begin() + cur_device_max_num,
                       it->second.end());
      if (it->second.empty()) {
        max_machine_id = machine_id - 1;
      } else {
        max_machine_id = machine_id;
      }
      break;
    } else {
      device_cnt += cur_device_num;
    }
  }
  CHECK_NE(max_machine_id, -1);
  FOR_EACH(it, sorted_machine_ids_) {
    if (*it > max_machine_id) {
      sorted_machine_ids_.erase(it, sorted_machine_ids_.end());
      break;
    }
  }
  EraseIf<int64_t, std::vector<int64_t>>(
      &machine_id2sorted_dev_phy_ids_,
      [&](HashMap<int64_t, std::vector<int64_t>>::iterator it) {
        return it->first > max_machine_id;
      });
  parallel_num_ = max_device_num;
}

void ParallelDesc::RemoveInvalidDevice(const std::string& op_name) {
  for (int64_t machine_id : sorted_machine_ids_) {
    auto& sorted_dev_ids = machine_id2sorted_dev_phy_ids_.at(machine_id);
    auto bound_it = std::lower_bound(
        sorted_dev_ids.begin(), sorted_dev_ids.end(),
        JobDesc::Singleton()->resource().device_num_per_machine());
    if (bound_it == sorted_dev_ids.end()) {
      continue;
    } else {
      for (auto it = bound_it; it != sorted_dev_ids.end(); ++it) {
        LOG_IF(WARNING, op_name != "")
            << op_name << " use invalid device_id " << *it;
      }
      sorted_dev_ids.erase(bound_it, sorted_dev_ids.end());
    }
  }
  ClearUp();
}

bool ParallelDesc::Equal(const ParallelDesc& rhs) const {
  return policy_ == rhs.policy_
         && sorted_machine_ids_ == rhs.sorted_machine_ids_
         && machine_id2sorted_dev_phy_ids_
                == rhs.machine_id2sorted_dev_phy_ids_;
}

void ParallelDesc::ClearUp() {
  EraseIf<int64_t, std::vector<int64_t>>(
      &machine_id2sorted_dev_phy_ids_,
      [](HashMap<int64_t, std::vector<int64_t>>::iterator it) {
        return it->second.empty();
      });
  sorted_machine_ids_.clear();
  parallel_num_ = 0;
  for (auto& pair : machine_id2sorted_dev_phy_ids_) {
    sorted_machine_ids_.push_back(pair.first);
    SortAndRemoveDuplication(&(pair.second));
    parallel_num_ += pair.second.size();
  }
  SortAndRemoveDuplication(&sorted_machine_ids_);
}

std::tuple<int32_t, int32_t> GetPartIdAndPartNumFromParallelCtx(
    const ParallelContext* parallel_ctx) {
  if (parallel_ctx->policy() == kDataParallel) {
    return std::make_tuple(0, 1);
  } else if (parallel_ctx->policy() == kModelParallel) {
    return std::make_tuple(parallel_ctx->parallel_id(),
                           parallel_ctx->parallel_num());
  } else {
    UNEXPECTED_RUN();
  }
}

}  // namespace oneflow
