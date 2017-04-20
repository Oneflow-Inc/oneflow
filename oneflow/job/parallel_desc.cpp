#include "job/parallel_desc.h"

namespace oneflow {
ParallelDesc::ParallelDesc(const ParallelConf& user_conf) {
  policy_ = user_conf.policy();
  device_type_ = JobDesc::Singleton().resource().device_type();
  
  // add a machine id to set, and add a device id to the map[machine_id]' vector
  for (int64_t i = 0; i < user_conf.devices_size(); i ++)
  {
    std::string device_name = user_conf.devices(i);
    int64_t delimiter_pos = device_name.find(":");
    CHECK(delimiter_pos != std::string::npos);

    std::string machine_name = device_name.substr(0, delimiter_pos);
    std::string device_id_str = device_name.substr(delimiter_pos,std::string::npos);
    uint64_t machine_id = IDMgr::Singleton().MachineID4MachineName(machine_name);
    sorted_machine_ids_.push_back(machine_id);

    if (device_id_str == "disk") {
      continue;
    }
    uint64_t device_id = 0;
    try {
      device_id = std::stoll(device_id_str);
    }
    catch (std::exception& e)
    {
      LOG(FATAL) << "error input:" << device_name << " (device id is not a integer or 'disk')";
    }
    machine_id2sorted_device_phy_ids_[machine_id].push_back(device_id);	
  }

  // Duplicate  and sort the container by ascending order
  std::sort(sorted_machine_ids_.begin(), sorted_machine_ids_.end());
  sorted_machine_ids_.erase(std::unique(sorted_machine_ids_.begin(), sorted_machine_ids_.end()), sorted_machine_ids_.end());
  for (auto it = machine_id2sorted_device_phy_ids_.begin(); it != machine_id2sorted_device_phy_ids_.end(); ++it)
  {
    int64_t device_ids_size_before_duplicate = it->second.size();
    std::sort( it->second.begin(),it->second.end() );
    it->second.erase(std::unique(it->second.begin(),it->second.end()),it->second.end());
    CHECK(device_ids_size_before_duplicate == it->second.size());
  }
}

} // namespace oneflow
