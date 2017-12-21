#ifndef ONEFLOW_CORE_MEMORY_MEMORY_DEVICE_H_
#define ONEFLOW_CORE_MEMORY_MEMORY_DEVICE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

class MemoryDevice final {
 public:
  explicit MemoryDevice(uint32_t machine_id, const MemoryCase& mem_case)
      : machine_id_(machine_id), mem_case_(mem_case) {}
  MemoryDevice(const MemoryDevice& mem_device) = default;

  const MemoryCase& mem_case() const { return mem_case_; }
  int64_t machine_id() const { return machine_id_; }
  int64_t device_id() const {
    return mem_case().has_device_cuda_mem()
               ? mem_case().device_cuda_mem().device_id()
               : 0;
  }
  bool IsGpuMem() const { return mem_case().has_device_cuda_mem(); }
  bool IsCpuMem() const { return !mem_case().has_device_cuda_mem(); }

  bool operator==(const MemoryDevice& mem_dev) const {
    return mem_dev.device_id() == device_id()
           && mem_dev.machine_id() == machine_id();
  }

 private:
  int64_t machine_id_;
  MemoryCase mem_case_;
};

}  // namespace oneflow

namespace std {

template<>
struct hash<oneflow::MemoryDevice> final {
  hash() = default;
  std::size_t operator()(const oneflow::MemoryDevice& mem_device) const {
    return mem_device.device_id() * INT32_MAX + mem_device.machine_id();
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_MEMORY_MEMORY_DEVICE_H_
