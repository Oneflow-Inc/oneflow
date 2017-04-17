#ifndef _DEVICE_DEVICE_DESCRIPTOR_H_
#define _DEVICE_DEVICE_DESCRIPTOR_H_

#include <cstdint>
#include <string>
#include "device/device_alternate.h"

/*
DeviceDescriptor contains all the hardware spec relating to a particular device. 
*/
namespace oneflow {
class DeviceDescriptor {
 public:
  explicit DeviceDescriptor(int32_t physical_id);
  ~DeviceDescriptor() {}
  std::string name() const;
  int32_t physical_id() const;
  int32_t compute_capability_major() const;
  int32_t compute_capability_minor() const;
  int32_t multi_processor_count() const;
  int32_t cuda_cores_per_multi_processor() const;
  int32_t total_cuda_cores() const;
  int32_t clock_rate() const; // MHz
  int32_t max_threads_per_multi_processor() const;
  int32_t warp_size() const;
  int32_t max_threads_per_block() const;
  void max_threads_dim(int32_t *d0, int32_t *d1, int32_t *d2) const;
  void max_grid_size(int32_t *s0, int32_t *s1, int32_t *s2) const;
  int32_t concurrent_kernerls() const;
  int32_t async_engine_count() const;
  int32_t memory_bus_width() const;
  size_t total_global_mem() const; // in bytes
  int32_t memory_clock_rate() const; // in MHz
  void SetCurrentDevice(int32_t device_id);

 private:
  std::string name_;
  int32_t physical_id_;
  int32_t compute_capability_major_;
  int32_t compute_capability_minor_;

  int32_t multi_processor_count_;
  int32_t cuda_cores_per_multi_processor_;
  int32_t total_cuda_cores_;
  int32_t clock_rate_;
  int32_t max_threads_per_multi_processor_;
  int32_t warp_size_;
  int32_t max_threads_per_block_;
  int32_t max_threads_dim_[3];
  int32_t max_grid_size_[3];

  int32_t concurrent_kernels_;
  int32_t async_engine_count_;
  size_t total_global_mem_;
  int32_t memory_clock_rate_;
  int32_t memory_bus_width_;

  void GetDeviceProperties();

  DeviceDescriptor(const DeviceDescriptor& other) = delete;
  DeviceDescriptor& operator=(const DeviceDescriptor& other) = delete;
};
inline int32_t DeviceDescriptor::physical_id() const {
  return physical_id_;
}
inline int32_t DeviceDescriptor::compute_capability_major() const {
  return compute_capability_major_;
}
inline int32_t DeviceDescriptor::compute_capability_minor() const {
  return compute_capability_minor_;
}
inline int32_t DeviceDescriptor::multi_processor_count() const {
  return multi_processor_count_;
}
inline int32_t DeviceDescriptor::cuda_cores_per_multi_processor() const {
  return cuda_cores_per_multi_processor_;
}
inline int32_t DeviceDescriptor::total_cuda_cores() const {
  return total_cuda_cores_;
}
// MHz
inline int32_t DeviceDescriptor::clock_rate() const {
  return clock_rate_;
}
inline int32_t DeviceDescriptor::max_threads_per_multi_processor() const {
  return max_threads_per_multi_processor_;
}
inline int32_t DeviceDescriptor::warp_size() const {
  return warp_size_;
}
inline int32_t DeviceDescriptor::max_threads_per_block() const {
  return max_threads_per_block_;
}
inline void DeviceDescriptor::max_threads_dim(
  int32_t *d0, int32_t *d1, int32_t *d2) const {
  *d0 = max_threads_dim_[0];
  *d1 = max_threads_dim_[1];
  *d2 = max_threads_dim_[2];
}
inline void DeviceDescriptor::max_grid_size(
  int32_t *s0, int32_t *s1, int32_t *s2) const {
  *s0 = max_grid_size_[0];
  *s1 = max_grid_size_[1];
  *s2 = max_grid_size_[2];
}
inline int32_t DeviceDescriptor::concurrent_kernerls() const {
  return concurrent_kernels_;
}
inline int32_t DeviceDescriptor::async_engine_count() const {
  return async_engine_count_;
}
// in bytes
inline size_t DeviceDescriptor::total_global_mem() const {
  return total_global_mem_;
}
// in MHz
inline int32_t DeviceDescriptor::memory_clock_rate() const {
  return memory_clock_rate_;
}
inline int32_t DeviceDescriptor::memory_bus_width() const {
  return memory_bus_width_;
}

inline int32_t GetDriverVersion() {
  int32_t driver_version;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  return driver_version;
}
inline int32_t GetRuntimeVersion() {
  int32_t runtime_version;
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  return runtime_version;
}
inline int32_t GetDeviceCount() {
  int32_t device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  return device_count;
}
inline int32_t GetCurrentDevice() {
  int32_t device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  return device_id;
}
inline void SetCurrentDevice(int32_t device_id) {
  CUDA_CHECK(cudaSetDevice(device_id));
  return;
}
}  // namespace oneflow
#endif  // _DEVICE_DEVICE_DESCRIPTOR_H_
