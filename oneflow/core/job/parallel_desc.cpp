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
  for (int64_t i = 0; i < user_conf.device_set().device_name_size(); ++i) {
    const std::string& device_name = user_conf.device_set().device_name(i);
    std::pair<std::string, std::string> machine_name_device_id =
        ParseDeviceNameConf(device_name);
    std::string machine_name = machine_name_device_id.first;
    std::string device_id_str = machine_name_device_id.second;
    int64_t machine_id =
        IDMgr::Singleton()->MachineID4MachineName(machine_name);
    sorted_machine_ids_.push_back(machine_id);
    // if the device_name format is "machine_xxx:0-3", add device_id {0,1,2,3}
    int64_t to_symbol_pos = device_id_str.rfind("-");
    if (device_id_str == "persistence") {
      machine_id2sorted_device_phy_ids_[machine_id] = {};
      device_type_ = DeviceType::kCPU;
    } else if (to_symbol_pos == std::string::npos) {
      int64_t device_id = oneflow_cast<int64_t>(device_id_str);
      machine_id2sorted_device_phy_ids_[machine_id].push_back(device_id);
    } else {
      int64_t begin_device_id =
          oneflow_cast<int64_t>(device_id_str.substr(0, to_symbol_pos));
      int64_t end_device_id =
          oneflow_cast<int64_t>(device_id_str.substr(to_symbol_pos + 1));
      CHECK_LT(begin_device_id, end_device_id);
      for (int64_t i = begin_device_id; i <= end_device_id; ++i) {
        machine_id2sorted_device_phy_ids_[machine_id].push_back(i);
      }
    }
  }
  SortAndRemoveDuplication(&sorted_machine_ids_);
  for (auto& pair : machine_id2sorted_device_phy_ids_) {
    SortAndRemoveDuplication(&(pair.second));
  }
  parallel_num_ = 0;
  for (auto const& pair : machine_id2sorted_device_phy_ids_) {
    parallel_num_ += pair.second.size();
  }
}

void ParallelDesc::RemoveNeedlessDevice(int32_t max_device_num) {
  if (max_device_num >= parallel_num_) { return; }
  int32_t device_cnt = 0;
  int64_t max_machine_id = -1;
  for (int64_t machine_id : sorted_machine_ids_) {
    auto it = machine_id2sorted_device_phy_ids_.find(machine_id);
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
  for (auto it = sorted_machine_ids_.begin(); it != sorted_machine_ids_.end();
       ++it) {
    if (*it > max_machine_id) {
      sorted_machine_ids_.erase(it, sorted_machine_ids_.end());
      break;
    }
  }
  EraseIf<int64_t, std::vector<int64_t>>(
      &machine_id2sorted_device_phy_ids_,
      [&](HashMap<int64_t, std::vector<int64_t>>::iterator it) {
        return it->first > max_machine_id;
      });
  parallel_num_ = max_device_num;
}

std::string ParallelDesc::VisualStr() const {
  std::stringstream ss;
  ss << "{policy:";
  if (policy_ == kDataParallel) {
    ss << "DataParallel";
  } else {
    ss << "ModelParallel";
  }
  ss << "}{device_type:";
  if (device_type_ == kGPU) {
    ss << "GPU";
  } else {
    ss << "CPU";
  }
  ss << "}{machine_id2sorted_device_phy_ids:";
  for (int64_t machine_id : sorted_machine_ids_) {
    ss << "{" << machine_id << ":[";
    for (int64_t device_phy_id :
         machine_id2sorted_device_phy_ids_.at(machine_id)) {
      ss << device_phy_id << ",";
    }
    ss << "]}";
  }
  ss << "}";
  return ss.str();
}

std::string GetMachineNameFromDeviceName(const std::string& device_name) {
  return ParseDeviceNameConf(device_name).first;
}

}  // namespace oneflow
