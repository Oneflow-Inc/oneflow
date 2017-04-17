#include "context/machine_descriptor.h"
#include "proto/oneflow.pb.h"
#include <unistd.h>

namespace oneflow {
MachineDescriptor::MachineDescriptor(const oneflow::SolverProto& solver) {
  // FIXME(jiyuan): set the thread num

  CHECK(solver.has_machine_id());
  machine_id_ = solver.machine_id();

  total_cpu_cores_ = sysconf(_SC_NPROCESSORS_ONLN);

  total_host_mem_ = sysconf(_SC_PHYS_PAGES);

  // All the installed devices, may including devices not allocated to current
  // job.
  // FIXME(jiyuan): to run the code on machine without GPUs, we temporally
  // disable the queries of CUDA devices. Please enable it in official code.
  // device_count_ = GetDeviceCount();
  // driver_version_ = GetDriverVersion();
  // runtime_version_ = GetRuntimeVersion();
  for (int32_t physical_id = 0; physical_id < device_count_; ++physical_id) {
    SetCurrentDevice(physical_id);
    std::unique_ptr<DeviceDescriptor> descriptor_ptr(nullptr);
    descriptor_ptr.reset(new DeviceDescriptor(physical_id));
    device_descriptors_.push_back(std::move(descriptor_ptr));
  }
}
int32_t MachineDescriptor::machine_id() const {
  return machine_id_;
}
int32_t MachineDescriptor::total_cpu_cores() const {
  return total_cpu_cores_;
}
int32_t MachineDescriptor::total_thread_num() const {
  return total_thread_num_;
}
size_t MachineDescriptor::total_host_mem() const {
  return total_host_mem_;
}
int32_t MachineDescriptor::device_count() const {
  return device_count_;
}
int32_t MachineDescriptor::device_thread_num() const {
  return device_thread_num_;
}
int32_t MachineDescriptor::driver_version() const {
  return driver_version_;
}
int32_t MachineDescriptor::runtime_version() const {
  return runtime_version_;
}
const std::unique_ptr<DeviceDescriptor>&
MachineDescriptor::device_descriptor(int32_t physical_id)
  const {
  CHECK_GE(physical_id, 0);
  CHECK_LT(physical_id, device_count_);
  return device_descriptors_[physical_id];
}
}  // namespace oneflow 
