#include "job/parallel_desc.h"
#include <algorithm>

namespace oneflow {

ParallelDesc::ParallelDesc(const ParallelConf& user_conf) {
  policy_ = user_conf.policy();
  device_type_ = JobDesc::Singleton().resource().device_type();
  for (int64_t i = 0; i < user_conf.device_set().device_name_size(); ++i){
    const std::string& device_name = user_conf.device_set().device_name(i);
    int64_t delimiter_pos = device_name.find(":");
    CHECK_NE(delimiter_pos, std::string::npos);

    std::string machine_name = device_name.substr(0, delimiter_pos);
    std::string device_id_str = device_name.substr(delimiter_pos);
    uint64_t machine_id =
        IDMgr::Singleton().MachineID4MachineName(machine_name);
    sorted_machine_ids_.push_back(machine_id);
    
    if (device_id_str == "disk") {
      continue;
    }
    uint64_t device_id = StoullOrDie(device_id_str);
    machine_id2sorted_device_phy_ids_[machine_id].push_back(device_id);	
  }
  SortAndRemoveDuplication(&sorted_machine_ids_);
  for (auto&pair : machine_id2sorted_device_phy_ids_) {
    SortAndRemoveDuplication(&(pair.second));
  }
}

} // namespace oneflow
