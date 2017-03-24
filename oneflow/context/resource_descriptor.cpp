#include "context/resource_descriptor.h"
#include <glog/logging.h>
#include "oneflow.pb.h"
#include "proto_io.h"

namespace oneflow {
ResourceDescriptor::ResourceDescriptor(const std::string& resource_name) :
  machine_num_(0), device_num_per_machine_(0) {
  oneflow::Resource resource;
  oneflow::ReadProtoFromTextFileOrDie(resource_name, &resource);
  machine_num_ = resource.machine_size();
  for (int32_t mid = 0; mid < machine_num_; ++mid) {
    oneflow::Machine machine = resource.machine(mid);
    MachineInfo machine_info;
    machine_info.name = machine.name();
    machine_info.port = machine.port();
    if (machine.has_device_set()) {
      // It is possible that there are no devices on a machine
      int32_t device_num = machine.device_set().device_id_size();
      if (device_num_per_machine_ == 0) {
        // Let |device_num_per_machine_| be the device number of the first machine
        device_num_per_machine_ = device_num;
      }
      else {
        // Ensure the other machine has the same number of devices as the first one
        CHECK_EQ(device_num, device_num_per_machine_)
          << "No equal number of devices on the machines";
      }

      for (int32_t did = 0; did < device_num; ++did) {
        machine_info.device_ids.push_back(machine.device_set().device_id(did));
        machine_info.physical2local.insert({ machine_info.device_ids.back(), did });
      }
    }
    machine_infos_.push_back(machine_info);
  }
}
int32_t ResourceDescriptor::machine_num() const {
  return machine_num_;
}
int32_t ResourceDescriptor::device_num_per_machine() const {
  return device_num_per_machine_;
}
int32_t ResourceDescriptor::total_device_num() const {
  return machine_num_ * device_num_per_machine_;
}
std::string ResourceDescriptor::machine_name(int32_t id) const {
  CHECK_GE(id, 0);
  CHECK_LT(id, machine_num_);
  return machine_infos_[id].name;
}
std::string ResourceDescriptor::machine_port(int32_t id) const {
  CHECK_GE(id, 0);
  CHECK_LT(id, machine_num_);
  return machine_infos_[id].port;
}
std::vector<int32_t> ResourceDescriptor::machine_device_ids(int32_t id) const {
  CHECK_GE(id, 0);
  CHECK_LT(id, machine_num_);
  return machine_infos_[id].device_ids;
}
int32_t ResourceDescriptor::local_from_physical(
  int32_t machine_id, int32_t physical_id) const {
  CHECK_GE(machine_id, 0);
  CHECK_LT(machine_id, machine_num_);
  auto it = machine_infos_[machine_id].physical2local.find(physical_id);
  CHECK(it != machine_infos_[machine_id].physical2local.end());
  return it->second;
}
ResourceDescriptor::~ResourceDescriptor() {
}
}  // namespace oneflow
