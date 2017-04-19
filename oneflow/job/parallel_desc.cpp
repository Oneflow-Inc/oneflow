#include "job/parallel_desc.h"

namespace oneflow {
inline bool isInt64_t(const std::string & s)
{
	if (s.empty() || ((!isdigit(s[0])) && (s[0] != '-') && (s[0] != '+'))) return false;

	char * p;
	std::strtoll(s.c_str(), &p, 10);
	return (*p == 0);
}

ParallelDesc::ParallelDesc(const ParallelConf& user_conf) {
	policy_ = user_conf.policy();
	device_type_ = JobDesc::Singleton().resource().device_type();
	
	// add a machine id to set, and add a device id to the map[machine_id]' vector
	std::set<int64_t> machine_ids;
	for (int i = 0; i < user_conf.devices_size(); i ++)
	{
		std::string device_name = user_conf.devices(i);

		if (device_name.find(":") == std::string::npos)
		{
			LOG(FATAL) << "error input: " << device_name << " (didn't contain ':')";
			continue;
		}
		std::string machine_name = device_name.substr(0,device_name.find(":"));
		std::string device_id_str = device_name.substr(device_name.find(":"),std::string::npos);
		int64_t machine_id = IDMgr::Singleton().MachineID4MachineName(machine_name);
		machine_ids.insert(machine_id);

		if (device_id_str == "disk") {
			continue;
		}
		if (!isInt64_t(device_id_str))
		{
			LOG(FATAL) << "error input:" << device_name << " (device id is not a integer or 'disk')";
			continue;
		}
		int64_t device_id = std::stoll(device_id_str);
		if (machine_id2sorted_device_phy_ids_.find(machine_id) == machine_id2sorted_device_phy_ids_.end())
		{
			machine_id2sorted_device_phy_ids_.insert(std::pair<int64_t,std::vector<int64_t>>(machine_id,std::vector<int64_t>()));
		}

		std::vector<int64_t>* p = & machine_id2sorted_device_phy_ids_.find(machine_id)->second;
		
		bool isRepeated = false;
		for (std::vector<int64_t>::iterator it = p->begin(); it != p->end(); ++it)
		{
			if (*it == device_id)
			{
				LOG(FATAL) << "Repeated value: " << device_name;
				isRepeated = true;
			}
		}

		if (!isRepeated)
		{
			p->push_back(device_id);
		}
		
	}

	// Duplicate  and sort the container by ascending order
	std::copy(machine_ids.begin(),machine_ids.end(),std::back_inserter(sorted_machine_ids_));
	std::sort(sorted_machine_ids_.begin(), sorted_machine_ids_.end());
	HashMap<int64_t, std::vector<int64_t>>::iterator it;
	for (it = machine_id2sorted_device_phy_ids_.begin(); it != machine_id2sorted_device_phy_ids_.end(); ++it)
	{
		std::sort( it->second.begin(),it->second.end() );
	}
}

} // namespace oneflow
