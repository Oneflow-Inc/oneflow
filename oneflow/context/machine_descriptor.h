#ifndef _CONTEXT_MACHINE_DESCRIPTOR_H_
#define _CONTEXT_MACHINE_DESCRIPTOR_H_

#include <cstdint>
#include <vector>
#include <memory>
#include <glog/logging.h>
#include "device/device_descriptor.h"

/*
Describes the available resources at current machine. We assume all the machines
have the same hardware configuration. This assumption simplify the design and 
implementation. Asymmetric hardware spec will be supported in the future. 
*/
namespace caffe {
class SolverProto;
class MachineDescriptor {
public:
  explicit MachineDescriptor(const caffe::SolverProto& solver);
  ~MachineDescriptor() {}

  int32_t machine_id() const;
  int32_t total_cpu_cores() const;
  size_t total_host_mem() const;
  // Return the number of GPU devices installed on this machine, may be larger 
  // than the number of devices allocated to this job. To get the number of 
  // devices for this job, please use ResourceDescriptor.device_num_per_machine
  int32_t device_count() const;
  int32_t driver_version() const;
  int32_t runtime_version() const;
  const std::unique_ptr<DeviceDescriptor>& device_descriptor(
    int32_t physical_id) const;

private:
  int32_t machine_id_;
  int32_t total_cpu_cores_;
  size_t total_host_mem_;
  int32_t device_count_;
  int32_t driver_version_;
  int32_t runtime_version_;
  std::vector<std::unique_ptr<DeviceDescriptor>> device_descriptors_;

  MachineDescriptor(const MachineDescriptor& other) = delete;
  MachineDescriptor& operator=(const MachineDescriptor& other) = delete;
};
}
#endif  // _CONTEXT_MACHINE_DESCRIPTOR_H_