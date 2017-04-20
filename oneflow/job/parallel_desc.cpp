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
    if (delimiter_pos == std::string::npos)
    {
      LOG(FATAL) << "error input: " << device_name << " (didn't contain ':')";
      continue;
    }
    std::string machine_name = device_name.substr(0, delimiter_pos);
    std::string device_id_str = device_name.substr(delimiter_pos,std::string::npos);
    int64_t machine_id = IDMgr::Singleton().MachineID4MachineName(machine_name);
    sorted_machine_ids_.push_back(machine_id);

    if (device_id_str == "disk") {
      continue;
    }
    try {
      int64_t device_id = std::stoll(device_id_str);
      machine_id2sorted_device_phy_ids_[machine_id].push_back(device_id);	
    }
    catch (std::exception& e)
    {
      LOG(FATAL) << "error input:" << device_name << " (device id is not a integer or 'disk')";
    }
  }

  // Duplicate  and sort the container by ascending order
  std::sort(sorted_machine_ids_.begin(), sorted_machine_ids_.end());
  sorted_machine_ids_.erase(std::unique(sorted_machine_ids_.begin(), sorted_machine_ids_.end()), sorted_machine_ids_.end());
  HashMap<int64_t, std::vector<int64_t>>::iterator it;
  for (it = machine_id2sorted_device_phy_ids_.begin(); it != machine_id2sorted_device_phy_ids_.end(); ++it)
  {
    int64_t device_ids_size_before_duplicate = it->second.size();
    std::sort( it->second.begin(),it->second.end() );
    it->second.erase(std::unique(it->second.begin(),it->second.end()),it->second.end());
    if (device_ids_size_before_duplicate > it->second.size())
    {
      LOG(FATAL) << "error input : repeated device id in :" << it->first << " machine";
    }
  }
}

} // namespace oneflow
