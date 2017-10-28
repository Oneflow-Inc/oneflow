#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

std::pair<std::string, std::string> ParseDeviceNameConf(
    const std::string& device_name) {
  int64_t delimiter_pos = device_name.rfind(":");
  CHECK_NE(delimiter_pos, std::string::npos);
  return {device_name.substr(0, delimiter_pos),
          device_name.substr(delimiter_pos + 1)};
}

ParallelDesc::ParallelDesc(const ParallelConf& user_conf) {
  policy_ = user_conf.policy();
  device_type_ = JobDesc::Singleton()->resource().device_type();
  for (const std::string& device_name : user_conf.device_name()) {
    auto machine_name_and_device_id_str = ParseDeviceNameConf(device_name);
    std::string mchn_name = machine_name_and_device_id_str.first;
    std::string device_id_str = machine_name_and_device_id_str.second;
    int64_t machine_id = IDMgr::Singleton()->MachineID4MachineName(mchn_name);
    sorted_machine_ids_.push_back(machine_id);
    if (device_id_str == "persistence" || device_id_str == "boxing") {
      device_type_ = DeviceType::kCPU;
      int64_t thrd_loc_id = IDMgr::Singleton()->GetThrdLocId(device_id_str);
      machine_id2sorted_thrd_loc_ids_[machine_id] = {thrd_loc_id};
      continue;
    }
    int64_t minus_pos = device_id_str.rfind("-");
    if (minus_pos == std::string::npos) {
      int64_t thrd_loc_id = oneflow_cast<int64_t>(device_id_str);
      machine_id2sorted_thrd_loc_ids_[machine_id] = {thrd_loc_id};
      continue;
    }
    int64_t min_id = oneflow_cast<int64_t>(device_id_str.substr(0, minus_pos));
    int64_t max_id = oneflow_cast<int64_t>(device_id_str.substr(minus_pos + 1));
    CHECK_LE(min_id, max_id);
    for (int64_t thrd_loc_id = min_id; thrd_loc_id <= max_id; ++thrd_loc_id) {
      machine_id2sorted_thrd_loc_ids_[machine_id].push_back(thrd_loc_id);
    }
  }
  SortAndRemoveDuplication(&sorted_machine_ids_);
  for (auto& pair : machine_id2sorted_thrd_loc_ids_) {
    SortAndRemoveDuplication(&(pair.second));
  }
  parallel_num_ = 0;
  for (const auto& pair : machine_id2sorted_thrd_loc_ids_) {
    parallel_num_ += pair.second.size();
  }
}

void ParallelDesc::RemoveNeedlessDevice(int32_t max_device_num) {
  if (max_device_num >= parallel_num_) { return; }
  int32_t device_cnt = 0;
  int64_t max_machine_id = -1;
  for (int64_t machine_id : sorted_machine_ids_) {
    auto it = machine_id2sorted_thrd_loc_ids_.find(machine_id);
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
      &machine_id2sorted_thrd_loc_ids_,
      [&](HashMap<int64_t, std::vector<int64_t>>::iterator it) {
        return it->first > max_machine_id;
      });
  parallel_num_ = max_device_num;
}

bool ParallelDesc::Equal(const ParallelDesc& rhs) const {
  return policy_ == rhs.policy_ && device_type_ == rhs.device_type_
         && sorted_machine_ids_ == rhs.sorted_machine_ids_
         && machine_id2sorted_thrd_loc_ids_
                == rhs.machine_id2sorted_thrd_loc_ids_;
}

}  // namespace oneflow
